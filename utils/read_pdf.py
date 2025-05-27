from langchain_community.document_loaders import PyPDFLoader

def read_pdf_file(pdf_path: str):
    """
    Reads a PDF file and returns its content as a list of documents.
    
    Args:
        pdf_path (str): Path to the PDF file.
        
    Returns:
        list: List of documents extracted from the PDF.
    """
    loader = PyPDFLoader(pdf_path)

    return loader.load() #documents