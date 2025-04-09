import os
from dotenv import load_dotenv
import logging
import base64
import io
import PIL.Image
import PIL.ImageOps

from transformers import AutoTokenizer

# from docling_core.types.doc import PictureItem, TextItem  # Adjusted based on actual module structure
from docling_core.types.doc.document import PictureDescriptionData
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import DocumentStream, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionLlamaStackApiOptions
from docling.datamodel.pipeline_options import granite_picture_description, smolvlm_picture_description

load_dotenv()
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

LLAMA_STACK_SERVER_CHAT_API=f'{os.getenv("LLAMA_STACK_SERVER")}/v1/inference/chat-completion'
VISION_MODEL="granite3.2-vision:2b"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256 #https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/discussions/1
VECTOR_DB_ID = "chat_documents"
  
def process_document(client, file_name, stream_bytes):
    # Read the PDF document using Docling
    # picture_description_options = smolvlm_picture_description
    # picture_description_options.prompt = """
    #         I am only interested in performance charts (bar, pie, histogram, boxplot, etc) that may appear in the image. 
    #         If it does NOT contain a chart, just give a short and concise description.
    #         But, If it does contain a chart of any kind please do your best to analyse it from a performance chart perspective and summarize the numerical or percentages represented in the whole chart series.
    #     """
    pdf_pipeline_options = PdfPipelineOptions(
        images_scale = 2.0,
        do_picture_description=True,
        generate_picture_images=True,
        enable_remote_services=True,
        picture_description_options= llama_stack_options(model=VISION_MODEL) # Doesn't work with llama-stack
        # picture_description_options= picture_description_options
    )
    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
    }

    document_stream = DocumentStream(name=file_name, stream=stream_bytes)
    doc = DocumentConverter(format_options=format_options).convert(source=document_stream).document
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
        print(f"=== Chunk #{i} ===")
        txt_tokens = len(tokenizer.tokenize(chunk.text))
        print(f"chunk.text ({txt_tokens} tokens):\n{repr(chunk.text)}")

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

# def get_text_items_before_picture(docling_document, picture_ref):
#     text_items_before_picture = []
#     found_picture = False

#     # Iterate through the groups in the document
    
#     for group in docling_document.groups:
#         # reset the text_items_before_picture for each group
#         text_items_before_picture = []
#         for item in group.children:
#             if isinstance(item, PictureItem) and item.get_ref().cref == picture_ref:
#                 # Stop searching once the Picture is found
#                 found_picture = True
#                 break
#             elif isinstance(item, TextItem):
#                 # Collect TextItems until the Picture is found
#                 text_items_before_picture.append(item)

#         if found_picture:
#             # Return the collected TextItems for the group containing the Picture
#             return text_items_before_picture

#     # If no Picture with the given ref is found, return an empty list
#     return []

# Doesen't work with llama-stack API yet!!!
def llama_stack_options(model: str):
    options = PictureDescriptionLlamaStackApiOptions(
        url=LLAMA_STACK_SERVER_CHAT_API,
        params=dict(
            model_id=model,
        ),
        prompt="""
            Here is an image. I am only interested in performance charts (bar, pie, histogram, boxplot, etc) that may appear in the image. 
            If it does NOT contain a chart, just give a short and concise description.
            But, If it does contain a chart of any kind please do your best to analyse it from a performance perspective and explain data points numerical or percentages represented in the chart.
        """,
        timeout=90,
    )
    return options

def process_document_images(client, docling_document):
    pictures = []
    tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)

    for picture in docling_document.pictures:
        image = picture.get_image(docling_document)
        if image:
            ref = picture.get_ref().cref
            print(f"Picture ref: {ref}")
            caption_text = picture.caption_text(doc=docling_document)
            print(f"Picture caption: {caption_text}")
            for annotation in picture.annotations:
                if not isinstance(annotation, PictureDescriptionData):
                    continue
                print(f"Annotations ({annotation.provenance}):{annotation.text}\n")       
                ser_tokens = len(tokenizer.tokenize(annotation.text))
                chunk = {
                    "content": annotation.text,
                    "mime_type": "text/plain",
                    "metadata": {
                        "document_id": ref,
                        "token_count": ser_tokens,
                        "source": docling_document.name,
                    }
                }
                pictures.append(chunk)
            # Get the TextItems that precede the Picture
            # text_items_before = get_text_items_before_picture(docling_document, ref)
            # preceding_text = " ".join(item.text for item in text_items_before)
            # print(f"Preceding text: {preceding_text}")

            # Get a text description of the image using a Vision Model
            # response = client.inference.chat_completion(
            #     model_id=VISION_MODEL,
            #     messages=[
            #         # {"role": "system", "content": "You're a helpful assistant specialized in explaining images."},
            #         {
            #             "role": "user",
            #             "content": [
            #                 {
            #                     "type": "text",
            #                     "text": f"""
            #                         Here is an image. I am only interested in performance charts (bar, pie, histogram, boxplot, etc) that may appear in the image. 
            #                         If it does NOT contain a chart, just ignore it! 
            #                         But, If it does contain a chart of any kind please do your best to analyse it from a performance chart perspective and summarize the numerical or percentages represented in it. 
            #                     """
            #                         # As a context for the analysis, I will provide you with the text that appears before the image in the document. Here it is: {preceding_text}
            #                 },
            #                 {
            #                     "type": "image", 
            #                     "image": {
            #                         "data": encode_image(image)
            #                     }
            #                 }
            #             ]
            #         },
            #     ],
            #     # temperature=0.0, 
            # )
            # print(response.completion_message.content)

            # ser_tokens = len(tokenizer.tokenize(response.completion_message.content))
            # ser_tokens = len(tokenizer.tokenize(response.completion_message.content))
            # chunk = {
            #     "content": response.completion_message.content,
            #     "mime_type": "text/plain",
            #     "metadata": {
            #         "document_id": ref,
            #         "token_count": ser_tokens,
            #         "source": docling_document.name,
            #     }
            # }
            # pictures.append(chunk)

    print(f"{len(pictures)} image descriptions created")
    return pictures