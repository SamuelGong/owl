# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2024 @ CAMEL-AI.org. All Rights Reserved. =========
import os
import sys

from dotenv import load_dotenv
from camel.models import ModelFactory
from camel.toolkits import (
    CodeExecutionToolkit,
    ExcelToolkit,
    ImageAnalysisToolkit,
    SearchToolkit,
    BrowserToolkit,
    FileWriteToolkit,
)
from camel.types import ModelPlatformType

from owl.utils import run_society
from camel.societies import RolePlaying
from camel.logger import set_log_level

import pathlib

base_dir = pathlib.Path(__file__).parent.parent
env_path = base_dir / "owl" / ".env"
load_dotenv(dotenv_path=str(env_path))

set_log_level(level="DEBUG")


def construct_society(question: str) -> RolePlaying:
    r"""Construct a society of agents based on the given question.

    Args:
        question (str): The task or question to be addressed by the society.

    Returns:
        RolePlaying: A configured society of agents ready to address the question.
    """

    # Create models for different components
    base_model_config = {
        "model_platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        "model_type": "ep-20250212105505-5zlbx",  # doubao-1.5-pro
        "api_key": os.getenv("ARK_API_KEY"),
        "url": os.getenv("ARK_API_BASE_URL"),
        "model_config_dict": {"temperature": 0.4, "max_tokens": 16384},  # max to set. Otherwise, it will fail
    }
    vision_model_config = {
        "model_platform": ModelPlatformType.OPENAI_COMPATIBLE_MODEL,
        "model_type": "ep-20241217114719-kcgs9",  # doubao-vision-pro
        "api_key": os.getenv("ARK_API_KEY"),
        "url": os.getenv("ARK_API_BASE_URL"),
        "model_config_dict": {"temperature": 0.4, "max_tokens": 4096},  # max to set. Otherwise, it will fail
    }
    models = {
        "user": ModelFactory.create(**base_model_config),
        "assistant": ModelFactory.create(**base_model_config),
        "browsing": ModelFactory.create(**vision_model_config),
        "planning": ModelFactory.create(**base_model_config),
        "image": ModelFactory.create(**vision_model_config),
    }

    # Configure toolkits
    tools = [
        *BrowserToolkit(
            # headless=False,  # Set to True for headless mode (e.g., on remote servers)
            headless=True,  # Set to True for headless mode (e.g., on remote servers)
            web_agent_model=models["browsing"],
            planning_agent_model=models["planning"],
        ).get_tools(),
        *CodeExecutionToolkit(sandbox="subprocess", verbose=True).get_tools(),
        *ImageAnalysisToolkit(model=models["image"]).get_tools(),
        SearchToolkit().search_duckduckgo,
        SearchToolkit().search_google,  # Comment this out if you don't have google search
        SearchToolkit().search_wiki,
        *ExcelToolkit().get_tools(),
        *FileWriteToolkit(output_dir="./").get_tools(),
    ]

    # Zhifeng: sanitize the tool schemas, otherwise model chatting fails
    for tool in tools:
        if hasattr(tool, "openai_tool_schema"):
            schema = tool.openai_tool_schema
            params_props = schema["function"]["parameters"]["properties"]

            keys_need_to_amend = []
            for k, v in params_props.items():
                if "anyOf" in v:
                    keys_need_to_amend.append(k)

                    for type in v["anyOf"]:
                        if type["type"] == "null":
                            continue
                        non_null_type = type["type"]

                    v["type"] = non_null_type
                    del v["anyOf"]

            required = schema["function"]["parameters"]["required"]
            for key in keys_need_to_amend:
                if key in required:
                    required.remove(key)

    # Configure agent roles and parameters
    user_agent_kwargs = {"model": models["user"]}
    assistant_agent_kwargs = {"model": models["assistant"], "tools": tools}

    # Configure task parameters
    task_kwargs = {
        "task_prompt": question,
        "with_task_specify": False,
    }

    # Create and return the society
    society = RolePlaying(
        **task_kwargs,
        user_role_name="user",
        user_agent_kwargs=user_agent_kwargs,
        assistant_role_name="assistant",
        assistant_agent_kwargs=assistant_agent_kwargs,
    )

    return society


def main():
    r"""Main function to run the OWL system with an example question."""
    # Example research question
    default_task = "Navigate to Amazon.com and identify one product that is attractive to coders. Please provide me with the product name and price. No need to verify your answer."
    # default_task = "f Eliud Kipchoge could maintain his record-making marathon pace indefinitely, how many thousand hours would it take him to run the distance between the Earth and the Moon its closest approach? Please use the minimum perigee value on the Wikipedia page for the Moon when carrying out your calculation. Round your result to the nearest 1000 hours and do not use any comma separators if necessary."

    # Override default task if command line argument is provided
    task = sys.argv[1] if len(sys.argv) > 1 else default_task

    # Construct and run the society
    society = construct_society(task)

    answer, chat_history, token_count = run_society(society)

    # Output the result
    print(f"\033[94mAnswer: {answer}\033[0m")


if __name__ == "__main__":
    main()
