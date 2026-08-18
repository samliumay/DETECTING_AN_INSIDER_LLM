
import os
from copy import deepcopy
from typing import Any
from google import genai
import requests

# Gemini client will be written.
class gemini_client():
    def __init__(self):
        print("")
        self.CLIENT = genai.Client()


    # Just a trial function for gemini. Thats all. Not a prod or resarch related functions.
    def trial(self, instructions: str) -> str:

        if not instructions:
            RuntimeError("You need to enter instructions to use the gemini_client.")


        answer = self.CLIENT.interactions.create(
            model="gemini-3.5-flash",
            input=instructions
        )

        return answer.output._text
