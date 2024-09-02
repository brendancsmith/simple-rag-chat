from tempfile import NamedTemporaryFile
from typing import List
import chainlit as cl

from chainlit.types import AskFileResponse
from langchain.document_loaders import PDFPlumberLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class DocumentManager():
    def __init__(self):
        self.index = 0

    def process_file(self, *, file: AskFileResponse) -> List[Document]:
        """Takes a Chainlit AskFileResponse, get the document and process and chunk
        it into a list of Langchain's Documents. Each Document has page_content and
        matadata fields. Supports PDF files only.

        Args:
            file (AskFileResponse): User's file input

        Raises:
            TypeError: when the file type is not pdf
            ValueError: when the PDF is not parseable

        Returns:
            List[Document]: chunked documents
        """

        print(file.path)

        if file.type == "application/pdf":
            with NamedTemporaryFile() as tmpfile:
                tmpfile.write(file.content)

                loader = PDFPlumberLoader(tmpfile.name)
                documents = loader.load()
        elif file.type == "text/plain":
            documents = [Document(page_content=file.content.decode())]
        else:
            raise TypeError("Only PDF and text files are supported")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        for doc in docs:
            doc.metadata["source"] = f"source_{self.index}"
            self.index += 1

        if docs is None:
            raise ValueError("File {} could not be parsed" % file.name)

        return docs

    async def process_docs(self, files):
        """Processes a list of files by calling the process_file method for each file.

        Args:
            files (list): A list of AskFileResponse objects representing the files to be processed.

        Returns:
            list: A list of Document objects representing the processed files.
        """

        docs = []
        for file in files:
            # Process and save data in the user session
            msg = cl.Message(content=f"Processing `{file.name}`...")
            await msg.send()
            docs += self.process_file(file=file)
            msg.content = f"`{file.name}` processed."
            await msg.update()

        return docs
