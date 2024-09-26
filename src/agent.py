import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)


from langgraph.graph import StateGraph
from src.utils import State, GraphInput, GraphOutput, GraphConfig
from src.nodes import *
from src.routers import *


def defining_nodes(workflow: StateGraph):
    workflow.add_node("content_generator", content_generator)
    workflow.add_node("content_critique", content_critique)
    workflow.add_node("translator", generate_translation)
    
    return workflow

def defining_edges(workflow: StateGraph):
    workflow.add_edge("content_generator","content_critique")
    workflow.add_conditional_edges(
        "content_critique",
        continue_with_feedback
    )
    workflow.add_edge("translator",END)
    

    return workflow


workflow = StateGraph(State, 
                      input = GraphInput,
                      output = GraphOutput,
                      config_schema = GraphConfig)

workflow.set_entry_point("content_generator")
workflow = defining_nodes(workflow = workflow)
workflow = defining_edges(workflow = workflow)

app = workflow.compile(
    )

if __name__ == '__main__':
    app = workflow.compile()
    config =  {
        "configurable": {
            "thread_id": 42,
            "model":"OpenAI : gpt-4o-mini",
            "content_temperature":0.22,
            "critique_temperature": 0,
            "critique_iterations": 2
        }
    }
    output = app.invoke(input = {"media_type":"carrousel","language":"spanish","topic":"regression","target_audience":"woman 35-45 living in spain","what_is_profile_about":"A psychologist who post content for education and self improvement"}, config = config) 
        
    print(output)
