"""Recursive chunker for splitting text into smaller chunks. uses LangChain's RecursiveCharacterTextSplitter to split text into chunks based on a specified chunk size and overlap."""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def recursive_chunk_text(
    text: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[str]:
    """Split text into smaller chunks using a recursive character text splitter.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to 1000.
        chunk_overlap (int, optional): The number of characters to overlap between chunks. Defaults to 200.

    Returns:
        list[str]: A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)
