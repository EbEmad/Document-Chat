import json
import os
from os.path import join
from tqdm.auto import tqdm
from typing import List, Optional, Literal
from datetime import datetime

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoModel
import torch
import chromadb
from chromadb import PersistentClient
from langchain.text_splitter import RecursiveCharacterTextSplitter
import PyPDF2
import docx
from pathlib import Path


class DocumentAssistant:
    def __init__(self, base_model_id="microsoft/Phi-4-mini-instruct"):
        self.base_model_id = base_model_id
        self.device = "cuda:0"
        self.torch_dtype = None
        
        # Initialize models and tokenizers
        self.llm_model, self.llm_tokenizer = self._load_llm_model()
        self.embedding_model, self.embedding_tokenizer = self._load_embedding_model()
        
        # Initialize ChromaDB client
        self.client = PersistentClient(path="chroma-db")
        
        # Dictionary to keep track of collections
        self.collections = {}

    def _load_llm_model(self):
        model=AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            device_map="auto",
            torch_dtype=self.torch_dtype
        )

        tokenizer=AutoTokenizer.from_pretrained(self.base_model_id)
        return model,tokenizer
    
    def _load_embedding_model(self):
        model_name = "intfloat/multilingual-e5-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        return model, tokenizer
    
    def embed_text(self, text):
        inputs = self.embedding_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.squeeze().cpu().numpy()
    

    def process_documents(self, file_path: str, collection_name: str = None) -> None:
        """
        Process documents (PDF, DOCX, TXT) and store embeddings in ChromaDB
        
        Args:
            file_path: Path to the document file
            collection_name: Optional custom name for the collection (if None, uses file stem)
        """
        # 1. Read the file
        text = self._read_file(file_path)
        
        # 2. Chunk the text
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 3. Create embeddings for chunks
        embeddings = [self.embed_text(chunk) for chunk in tqdm(chunks, desc="Creating embeddings")]
        
        # 4. Store in ChromaDB
        if collection_name is None:
            collection_name = Path(file_path).stem
            
        # Ensure collection name is valid
        collection = self.client.get_or_create_collection(name=collection_name)
        
        # 5. Add documents and embeddings to collection
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            collection.add(
                documents=[chunk],
                embeddings=[embedding.tolist()],
                metadatas=[{"chunk_id": idx, "source": file_path}],
                ids=[f"chunk_{idx}"]
            )
        
        # Store collection reference
        self.collections[collection_name] = {
            'file_path': file_path,
            'num_chunks': len(chunks)
        }
        
        print(f"Successfully processed and stored {len(chunks)} chunks from {file_path}")


    def remove_document(self,collection_name:str)->bool:
        """
        Remove a document collection from ChromaDB
        
        Args:
            collection_name: Name of the collection to remove
        
        Returns:
            bool: True if successfully removed, False otherwise
        """

        try:
            self.client.delete_collection(name=collection_name)

            if collection_name in self.collections:
                del self.collections[collection_name]
            return True
        except Exception as e:
            print(f"Error removing collection {collection_name}: {e}")
            return False
        
    def list_documnets(self)->dict:
        """
        List all processed documents
        
        Returns:
            dict: A dictionary of processed documents with their details
        """
        return self.collections
    
    def _read_file(self,file_path:str)->str:
        """
        Read different file formats and return text content
        """

        file_extension=Path(file_path).suffix.lower()

        if file_extension==".pdf":
            with open(file_path,'rb') as file:
                pdf_reader=PyPDF2.PdfReader(file)
                text=""
                for page in pdf_reader.pages:
                    text+=page.extract_text()
        elif file_extension=='.docx':
            doc=docx.Document(file_path)
            text="\n".join([paragraph.text for paragraph in doc.paragraphs])

        elif file_extension==".text":
            with open(file_path,'r',encoding='utf-8') as file:
                text=file.read()
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")

        return text
      
    def generate_prompt(self, question, context):
        """
        Generate a more general prompt for any kind of document
        """
        system_message = "\n".join([
            "You are a knowledgeable assistant. Your task is to provide accurate answers based on the provided context.",
            "Your response should be clear and concise, using the same language as the question.",
            "Make sure to reference specific information from the context when applicable.",
            "Focus on providing relevant information without adding unnecessary details.",
            "If the context doesn't contain enough information to answer the question, please indicate that."
        ])

        messages = [
            {
                "role": "system",
                "content": system_message
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nContext: {context}\n\nAnswer:"
            }
        ]

        return messages
    
    def generate_resp(self,messages):
        text=self.llm_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs=self.llm_tokenizer([text],return_tensors="pt").to(self.device)

        generated_ids = self.llm_model.generate(
            model_inputs.input_ids,
            max_new_tokens=1024,
            do_sample=False, top_k=None, temperature=None, top_p=None,
        )


     

