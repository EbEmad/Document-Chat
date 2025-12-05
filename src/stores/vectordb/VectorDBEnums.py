from enum import Enum


class VectorDBEnums(Enum):
    QDRANT= "QDRANT"
    PINECONE= "PINECONE"
    FAISS= "FAISS"
    


class DistanceMethodEnums(Enum):
    COSINE= "cosine"
    L2= "l2"
    DOT="dot"
