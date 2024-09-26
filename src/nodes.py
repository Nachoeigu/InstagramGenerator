import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from src.utils import State, GraphConfig, _get_model, ContentCreatorCarrouselStructuredOutput, ContentCreatorPictureStructuredOutput, ContentCreatorVideoStructuredOutput
from constants import CONTENT_GENERATOR, CONTENT_CRITIQUE, TRANSLATOR
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import uuid

def content_generator(state: State, config: GraphConfig) -> State:
    model = _get_model(config, key = 'content')
    
    if state['content_generations'] == []:
        model_input = [SystemMessage(content = CONTENT_GENERATOR), HumanMessage(content = f"The user requests a {state['media_type']} post focused on {state['topic']}. The content should be tailored to resonate with the {state['target_audience']}. The account is about {state['what_is_profile_about']}", id = uuid.uuid1())]
    else:
        model_input = [SystemMessage(content = CONTENT_GENERATOR), HumanMessage(content = f"The user requests a {state['media_type']} post focused on {state['topic']}. The content should be tailored to resonate with the {state['target_audience']}. The account is about {state['what_is_profile_about']}", id = uuid.uuid1())] + state['content_generations']

    media_type = state['media_type']

    if media_type == 'carrousel':
        model = model.with_structured_output(ContentCreatorCarrouselStructuredOutput)
    elif media_type == 'one picture':
        model = model.with_structured_output(ContentCreatorPictureStructuredOutput)
    elif media_type == 'video':
        model = model.with_structured_output(ContentCreatorVideoStructuredOutput)
    
    result = model.invoke(model_input)

    return {"content_generations": [AIMessage(content = str(result))]}

def content_critique(state: State, config: GraphConfig) -> State:
    content_generation = HumanMessage(content = state['content_generations'][-1].content)
    model = _get_model(config, key = 'critique')
    n_critiques = state.get("critique_iterations", 0)

    if n_critiques == 0:
        model_input = [SystemMessage(content = CONTENT_CRITIQUE.format(topic=state['topic'],target_audience=state['target_audience'], profile_about=state['what_is_profile_about'])), HumanMessage(content = content_generation.content, id = uuid.uuid4())]
    else:
        model_input = [SystemMessage(content = CONTENT_CRITIQUE.format(topic=state['topic'],target_audience=state['target_audience'], profile_about=state['what_is_profile_about'])), HumanMessage(content = content_generation.content, id = uuid.uuid4())] + state['critique_generations'] + [content_generation]
        
    result = model.invoke(model_input)

    return {'critique_generations': [result],
            'content_generations': [HumanMessage(content = result.content, id = uuid.uuid4())],
            'critique_iterations': n_critiques + 1}

def generate_translation(state: State, config: GraphConfig) -> State:
    model = _get_model(config)

    last_version = [msg for msg in state['content_generations'] if isinstance(msg, AIMessage)][-1].content
   
    messages = [
        SystemMessage(content = TRANSLATOR.format(target_language=state['language'])),
        HumanMessage(content = last_version, id = uuid.uuid4())
    ]
    
    result = model.invoke(messages)

    return {
        "translated_content": result
    }
