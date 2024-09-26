import os
from dotenv import load_dotenv
import sys

load_dotenv()
WORKDIR=os.getenv("WORKDIR")
os.chdir(WORKDIR)
sys.path.append(WORKDIR)

from typing import  Literal
from constants import *
from src.utils import State, GraphConfig
from langgraph.graph import END


def continue_with_feedback(state: State, config: GraphConfig) -> Literal['content_generator',END, 'translator']:
    if config['configurable'].get('critique_iterations', 5) < state['critique_iterations']:
        if state['language'] == 'english':
            return END
        else:
            return "translator"
    else:
        return 'content_generator'