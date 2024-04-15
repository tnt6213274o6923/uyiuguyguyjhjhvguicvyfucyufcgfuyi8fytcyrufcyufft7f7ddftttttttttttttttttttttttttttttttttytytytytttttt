import torch
import gc

# Function to clear GPU memory
def clear_gpu_memory():
    # Delete model and tensors if they are defined
    global model, inputs, input_ids
    if 'model' in globals():
        del model
    if 'inputs' in globals():
        del inputs
    if 'input_ids' in globals():
        del input_ids

    # Clear PyTorch cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Force Python's garbage collector to run
    gc.collect()

# Call the function to clear GPU memory
clear_gpu_memory()

import os
import sys

import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

"""
Helpers to support streaming generate output.
Borrowed from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/callbacks.py
"""

import gc
import traceback
from queue import Queue
from threading import Thread

import torch
import transformers

class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids, scores) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False

class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gentask():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)
            if self.c_callback:
                self.c_callback(ret)

        self.thread = Thread(target=gentask)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True

"""
A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        template_name = "alpaca"
        self.template = {
            "description": "Template used by Alpaca-LoRA.",
            "prompt_input": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n",
            "prompt_no_input": "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n",
            "response_split": "### Response:"
        }
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res

    def get_response(self, output: str) -> str:
        return output.split(self.template["response_split"])[1].strip()


if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


base_model = 'openthaigpt/openthaigpt-1.0.0-beta-7b-chat-ckpt-hf'
lora_weights = None
load_8bit = False
prompt_template = ""
server_name = "0.0.0.0"
share_gradio = True

prompter = Prompter(prompt_template)
tokenizer = LlamaTokenizer.from_pretrained(base_model)
if device == "cuda":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        load_in_8bit=load_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          torch_dtype=torch.float16,
      )
elif device == "mps":
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        device_map={"": device},
        torch_dtype=torch.float16,
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          device_map={"": device},
          torch_dtype=torch.float16,
      )
else:
    model = LlamaForCausalLM.from_pretrained(
        base_model, device_map={"": device}, low_cpu_mem_usage=True
    )
    if lora_weights:
      model = PeftModel.from_pretrained(
          model,
          lora_weights,
          device_map={"": device},
      )

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

if not load_8bit:
    model.half()  # seems to fix bugs for some users.

model.eval()
if torch.__version__ >= "2" and sys.platform != "win32":
    model = torch.compile(model)

def evaluate(
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
    repetition_penalty=1,
    no_repeat_ngram=0,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
        "early_stopping": True,
        "repetition_penalty":repetition_penalty,
        "no_repeat_ngram_size":no_repeat_ngram
    }


    if stream_output:
        # Stream the reply 1 token at a time.
        # This is based on the trick of using 'stopping_criteria' to create an iterator,
        # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

        def generate_with_callback(callback=None, **kwargs):
            kwargs.setdefault(
                "stopping_criteria", transformers.StoppingCriteriaList()
            )
            kwargs["stopping_criteria"].append(
                Stream(callback_func=callback)
            )
            with torch.no_grad():
                model.generate(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(
                generate_with_callback, kwargs, callback=None
            )

        with generate_with_streaming(**generate_params) as generator:
            for output in generator:
                # new_tokens = len(output) - len(input_ids[0])
                decoded_output = tokenizer.decode(output)

                if output[-1] in [tokenizer.eos_token_id]:
                    break

                yield prompter.get_response(decoded_output)
        return  # early return for stream_output

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield prompter.get_response(output)

def clearText():
    return ["","",""]

def example1():
    return ["ลดความอ้วนต้องทำอย่างไร",""]

def example2():
    return ["วางแผนเที่ยวในภูเก็ต แบบบริษัททัวร์","ภูเก็ต เป็นจังหวัดหนึ่งทางภาคใต้ของประเทศไทย และเป็นเกาะขนาดใหญ่ที่สุดในประเทศไทย อยู่ในทะเลอันดามัน จังหวัดที่ใกล้เคียงทางทิศเหนือ คือ จังหวัดพังงา ทางทิศตะวันออก คือ จังหวัดพังงา ทั้งเกาะล้อมรอบด้วยมหาสมุทรอินเดีย และยังมีเกาะที่อยู่ในอาณาเขตของจังหวัดภูเก็ตทางทิศใต้และตะวันออก การเดินทางเข้าสู่ภูเก็ตนอกจากทางเรือแล้ว สามารถเดินทางโดยรถยนต์ซึ่งมีเพียงเส้นทางเดียวผ่านทางจังหวัดพังงา โดยข้ามสะพานสารสินและสะพานคู่ขนาน คือ สะพานท้าวเทพกระษัตรีและสะพานท้าวศรีสุนทร เพื่อเข้าสู่ตัวจังหวัด และทางอากาศโดยมีท่าอากาศยานนานาชาติภูเก็ตรองรับ ท่าอากาศยานนี้ตั้งอยู่ทางทิศตะวันตกเฉียงเหนือของเกาะ"]

def example3():
    return ["เขียนบทความเกี่ยวกับ \"ประโยชน์ของโกจิเบอร์รี่\"",""]

def example4():
    return ["เขียนโค้ด","python pandas csv export"]

def example5():
    return ["x+30=100 x=?",""]

def example6():
    return ["แปลภาษาไทยเป็นอังกฤษ","กรุงเทพมหานคร เป็นเมืองหลวงและนครที่มีประชากรมากที่สุดของประเทศไทย เป็นศูนย์กลางการปกครอง การศึกษา การคมนาคมขนส่ง การเงินการธนาคาร การพาณิชย์ การสื่อสาร และความเจริญของประเทศ"]

with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🇹🇭 OpenThaiGPT 1.0.0-beta
        🇹🇭 OpenThaiGPT Version 1.0.0-beta is a Thai language 7B-parameter LLaMA v2 Chat model finetuned to follow Thai translated instructions and extend 24,554 Thai words vocabularies for turbo speed. For more information, please visit [the project's website](https://openthaigpt.aieat.or.th/) | [Github](https://github.com/OpenThaiGPT/openthaigpt).

        ## Examples
        """
    )
    with gr.Row():
        example1_button = gr.Button(value="ลดความอ้วนต้องทำอย่างไร")
        example2_button = gr.Button(value="วางแผนเที่ยวในภูเก็ต แบบบริษัททัวร์")
        example3_button = gr.Button(value="เขียนบทความ")
        example4_button = gr.Button(value="เขียนโค้ด")
        example5_button = gr.Button(value="คำนวณคณิตศาสตร์")
        example6_button = gr.Button(value="แปลภาษา")

    instbox = gr.components.Textbox(
            lines=2,
            label="Instruction",
            placeholder="คำสั่ง",
            value="ลดความอ้วนต้องทำอย่างไร"
        )
    inputbox = gr.components.Textbox(lines=2, label="Input", placeholder="คำถาม (ไม่จำเป็น)")
    streambox = gr.components.Checkbox(label="Stream output", value=True)
    button = gr.Button(value="Generate", variant="primary")

    with gr.Row():
        cancel = gr.Button(value="Stop / Cancel")
        clear = gr.Button(value="Clear")

    outputbox = gr.inputs.Textbox(
            lines=5,
            label="Output",
        )

    with gr.Accordion("Advanced Settings", open=False):
        tempbox = gr.components.Slider(
            minimum=0, maximum=1, value=0.1, info="อุณหภูมิ: พารามิเตอร์นี้ใช้ควบคุมความเสี่ยงในการสร้างข้อความของระบบ ถ้าตั้งค่าไว้สูง การสร้างข้อความจะเป็นลักษณะที่หลากหลายมากขึ้น ถ้าตั้งค่าไว้ต่ำ การสร้างข้อความจะมีลักษณะที่มีโครงสร้างแน่นอนมากขึ้น", label="Temperature"
        )
        toppbox = gr.components.Slider(
            minimum=0, maximum=1, value=0.75, info="nucleus sampling: พารามิเตอร์นี้ใช้เป็นวิธีการสุ่มตัวเลือกจากคำที่อาจจะถูกเลือกถัดไป ระบบจะสุ่มเลือกจากกลุ่มคำที่มีความน่าจะเป็นรวมกันสูงสุดถึง p%", label="Top p"
        )
        topkbox = gr.components.Slider(
            minimum=0, maximum=100, step=1, value=40, info="top-k sampling: พารามิเตอร์นี้ใช้เลือก k คำที่มีความน่าจะเป็นสูงสุดสำหรับคำถัดไป แล้วจึงสุ่มเลือกหนึ่งใน k คำนั้น", label="Top k"
        )
        beambox = gr.components.Slider(
            minimum=1, maximum=4, step=1, value=1, info="beam: จำนวนวิธีการสร้างข้อความโดยใช้คำหลายๆ ทางเลือกที่น่าจะเป็นที่สุดในแต่ละขั้นตอน การตั้งค่า Beam ที่สูงขึ้นจะทำให้สามารถสำรวจคำหลายทางเลือกมากขึ้น แต่จะเพิ่มการคำนวณและอาจจะไม่ทำให้ผลลัพธ์ดีขึ้นทุกครั้ง", label="Beams"
        )
        maxtokenbox = gr.components.Slider(
            minimum=1, maximum=4096, step=1, value=512, info="max_token: ความยาวของคำตอบ", label="Max tokens"
        )
        repetition_penalty_box = gr.components.Slider(
            minimum=1, maximum=1.99, step=0.01, value=1.2, info="repetition_penalty: ความรุนแรงในการลงโทษเมื่อตอบข้อความซ้ำ 1=ไม่ลงโทษ 1.99=ลงโทษสูงสุด", label="Repetition Penalty"
        )
        no_repeat_ngram_box = gr.components.Slider(
            minimum=0, maximum=30, step=0, value=4, info="no_repeat_ngram: การป้องกันการตอบข้อความซ้ำตามจำนวนตัวอักษร", label="No Repeat N-GRAM"
        )

    button_click_event = button.click(fn=evaluate, inputs=[instbox, inputbox, tempbox, toppbox, topkbox, beambox, maxtokenbox, streambox, repetition_penalty_box, no_repeat_ngram_box], outputs=outputbox)
    cancel.click(fn=None, inputs=None, outputs=None, cancels=[button_click_event])
    clear.click(fn=clearText, outputs=[instbox, inputbox, outputbox])

    example1_button.click(fn=example1, outputs=[instbox, inputbox])
    example2_button.click(fn=example2, outputs=[instbox, inputbox])
    example3_button.click(fn=example3, outputs=[instbox, inputbox])
    example4_button.click(fn=example4, outputs=[instbox, inputbox])
    example5_button.click(fn=example5, outputs=[instbox, inputbox])
    example6_button.click(fn=example6, outputs=[instbox, inputbox])

demo.queue().launch(server_name="0.0.0.0", share=share_gradio)
