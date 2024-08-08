from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
import base64
from io import BytesIO

from cognitive_bt_framework.src.sim.ai2_thor.ai2_thor_sim import AI2ThorSimEnv

model = 'OpenGVLab/InternVL2-8B'
system_prompt = '我是书生·万象，英文名是InternVL，是由上海人工智能实验室及多家合作单位联合开发的多模态大语言模型。'
image = load_image('https://raw.githubusercontent.com/open-mmlab/mmdeploy/main/tests/data/tiger.jpeg')
chat_template_config = ChatTemplateConfig('internvl-internlm2')
chat_template_config.meta_instruction = system_prompt

sim = AI2ThorSimEnv()
image = sim.get_context(1)[0]
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

task = "make a sandwich"
failure = "cannot pick up bread agent is already holding knife"

instruction = f"I am trying to {task} but failed due to {failure}, what do you see in my environment?"
prompt = f"""User:<image>\n{instruction} Falcon:"""

raw_image = Image.open(BytesIO(base64.b64decode(image)))
print('OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO')
pipe = pipeline(model, chat_template_config=chat_template_config,
                backend_config=TurbomindEngineConfig(session_len=8192))
print("PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
response = pipe(('describe this image', raw_image))