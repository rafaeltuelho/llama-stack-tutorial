import os
from dotenv import load_dotenv
import logging
import base64
import io
import PIL.Image
import PIL.ImageOps

from transformers import AutoTokenizer

from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VISION_MODEL="granite3.2-vision:2b"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256 #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/1
VECTOR_DB_ID = "chat_documents"

def register_vector_db(client):
    try:
        response = client.vector_dbs.retrieve(vector_db_id=VECTOR_DB_ID)
        if response:
            print(f"Vector db {VECTOR_DB_ID} already exists.")
            return
    except Exception as e:
        logger.error(f"Vector db {VECTOR_DB_ID} does not exist: {e}")
        response = client.vector_dbs.register(
            vector_db_id=VECTOR_DB_ID,
            embedding_model="all-MiniLM-L6-v2",
            embedding_dimension=384,
            provider_id="faiss",
        )
        if response:
            print(f"Vector db {VECTOR_DB_ID} registered.")
        else:
            return
  
def process_document(client, file_name, stream_bytes):
    # Read the PDF document using Docling
    register_vector_db(client)

    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=False,
        generate_picture_images=True,
    )
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
    }

    documentStream = DocumentStream(name=file_name, stream=stream_bytes)
    doc = DocumentConverter(format_options=format_options).convert(source=documentStream).document
    # Extract document text chunks
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    chunker = HybridChunker(
        tokenizer=tokenizer,  # instance or model name, defaults to "sentence-transformers/all-MiniLM-L6-v2"
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
        merge_peers=True,  # optional, defaults to True
    )

    chunk_iter = chunker.chunk(dl_doc=doc)
    chunks = list(chunk_iter)
    ser_chunks = []
    for i, chunk in enumerate(chunks):
        # print(f"=== Chunk #{i} ===")
        # txt_tokens = len(tokenizer.tokenize(chunk.text))
        # print(f"chunk.text ({txt_tokens} tokens):\n{repr(chunk.text)}")

        ser_txt = chunker.serialize(chunk=chunk)
        ser_tokens = len(tokenizer.tokenize(ser_txt))
        # print(f"chunker.serialize(chunk) ({ser_tokens} tokens):\n{repr(ser_txt)}")

        ser_chunks.append({
            "content": ser_txt,
            "mime_type": "text/plain",
            "metadata": {
                "document_id": doc.name,
                "token_count": ser_tokens,
                "source": doc.name,
            }
        })

    # print(f"chunks vectorized: {len(ser_chunks)}")
    # insert the chunks into the vector db
    client.vector_io.insert(vector_db_id=VECTOR_DB_ID, chunks=ser_chunks)

    # Process images
    pictures_descriptions = process_document_images(client, doc)
    # Insert image descriptions into the vector db
    client.vector_io.insert(vector_db_id=VECTOR_DB_ID, chunks=pictures_descriptions)



def encode_image(image: PIL.Image.Image, format: str = "png") -> str:
    image = PIL.ImageOps.exif_transpose(image) or image
    image = image.convert("RGB")

    buffer = io.BytesIO()
    image.save(buffer, format)
    encoding = base64.b64encode(buffer.getvalue()).decode("utf-8")
    # uri = f"data:image/{format};base64,{encoding}"
    # print(f"Image encoded: {uri}")
    print(f"\n\nImage encoded: [{encoding}]\n\n")
    return encoding

def process_document_images(client, docling_document):
    pictures = []
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    chunker = HybridChunker(
        tokenizer=tokenizer,  # instance or model name, defaults to "sentence-transformers/all-MiniLM-L6-v2"
        max_tokens=MAX_TOKENS,  # optional, by default derived from `tokenizer`
        merge_peers=True,  # optional, defaults to True
    )

    for picture in docling_document.pictures:
        ref = picture.get_ref().cref
        image = picture.get_image(docling_document)
        if image:
            # text = vision_model.invoke(vision_prompt, image=encode_image(image))
            # Get a text description of the image using a Vision Model
            response = client.inference.chat_completion(
                model_id=VISION_MODEL,
                messages=[
                    # {"role": "system", "content": "You're a helpful assistant specialized in explaining images."},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """
                                    If is there any text in the image, please describe it. 
                                    If there is a graph, please describe it. 
                                    If there is a chart, please describe it. 
                                    If there is a table, please describe it. 
                                    If there is a picture, please describe it. 
                                    If there is a diagram, please describe it. 
                                    If there is a map, please describe it. 
                                    If there is a screenshot, please describe it. 
                                    If there is a photo, please describe it. 
                                    If there is an illustration, please describe it. 
                                    If there is a drawing, please describe it. 
                                    If there is a painting, please describe it. 
                                    If there is a sculpture, please describe it. 
                                    If there is an object, please describe it.",
                                """
                            },
                            {
                                "type": "image", 
                                "image": {
                                    "data": encode_image(image)
                                }
                            }
                        ]
                    },
                ],
                # temperature=0.0, 
            )
            print(response.completion_message.content)

            ser_tokens = len(tokenizer.tokenize(response.completion_message.content))
            chunk = {
                "content": response.completion_message.content,
                "mime_type": "text/plain",
                "metadata": {
                    "document_id": ref,
                    "token_count": ser_tokens,
                    "source": docling_document.name,
                }
            }
            pictures.append(chunk)

    print(f"{len(pictures)} image descriptions created")
    return pictures