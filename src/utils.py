import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

import time
from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List, Literal
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_aws.chat_models import ChatBedrock
from constants import *
import re

class ContentCreatorVideoStructuredOutput(TypedDict):
    """
    The structured answer of the social media manager for video content
    """
    video_title: str = Field(description = "The title of the video, displayed before start it")
    video_description: str = Field(description = "The description of the post, below the content itself. Use hashtags and emojis if needed")
    text_in_video: str = Field(description = "A list with all the speech/text should be said or displayed, inside the video.")
    
class ContentCreatorCarrouselStructuredOutput(TypedDict):
    """
    The structured answer of the social media manager for carrousel content
    """
    n_pictures: int = Field(description = "The total images in the carrousel")
    text_each_image: List[str] = Field(description = "A list where each element is the text inside each picture in the carrousel.")
    description: str = Field(description = "The description of the post, below the carrousel. Use hashtags and emojis if needed")

class ContentCreatorPictureStructuredOutput(TypedDict):
    """
    The structured answer of the social media manager for picture content
    """
    text_in_image: List[str] = Field(description = "Place the text which should go inside the picture.")
    description: str = Field(description = "The description of the post, below the content itself. Use hashtags and emojis if needed")


class GraphConfig(BaseModel):
    """
    Initial configuration to trigger the AI system.

    Attributes:
    - qa_model: Select the model for the LLM. Options include 'openai', 'google', 'meta', or 'amazon'.
    - system_prompt: Select the prompt of your conversation.
    - temperature: Select the temperature for the model. Options range from 0 to 1.
    - using_summary_in_memory: If you want to summarize previous messages, place True. Otherwise, False.
    """
    model: Literal[*AVAILABLE_MODELS] = 'OpenAI: gpt-4o'
    content_temperature: float = 0.5
    critique_temperature: float = 0
    critique_iterations: int

class GraphInput(TypedDict):
    """
    The initial message that starts the AI system
    """
    media_type: Literal["carrousel", "one picture", "video"]
    language: Literal['english', 'spanish', 'portuguese', 'poland', 'french', 'german', 'italian', 'dutch','swedish', 'norwegian', 'danish', 'finnish', 'russian', 'chinese', 'japanese', 'korean','arabic', 'turkish', 'greek', 'hebrew']
    topic: str
    target_audience: str
    what_is_profile_about: str


class GraphOutput(TypedDict):
    """
    The output of the AI System
    """
    content_generations: List[str]

class State(TypedDict):
    media_type: Literal["carrousel", "one picture", "video"]
    language: Literal['english', 'spanish', 'portuguese', 'poland', 'french', 'german', 'italian', 'dutch','swedish', 'norwegian', 'danish', 'finnish', 'russian', 'chinese', 'japanese', 'korean','arabic', 'turkish', 'greek', 'hebrew']
    topic: str
    what_is_profile_about: str
    target_audience: str
    critique_iterations: int = 0
    translated_content: str
    content_generations: Annotated[List[AnyMessage], add_messages]
    critique_generations: Annotated[List[AnyMessage], add_messages]

def _get_model(config: GraphConfig, key=Literal['critique','content'], default='OpenAI: gpt-4o'):
    model = config['configurable'].get('model', 'OpenAI: gpt-4o')
    temperature = config['configurable'].get(f'{key}_temperature', 0.5)
    company, model_name = model.split(' : ')[0].strip().lower(), model.split(' : ')[-1].strip()

    if company == "openai":
        model = ChatOpenAI(temperature=temperature, model=model_name)
    elif company == "google":
        model = ChatGoogleGenerativeAI(temperature=temperature, model=model_name)
    elif company == 'groq':
        model = ChatGroq(temperature=temperature, model=model_name)
    elif company == 'amazon':
        model = ChatBedrock(model_id =model_name, model_kwargs = {'temperature':temperature})
    else:
        raise ValueError
    
    return model


