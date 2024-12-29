import transformers
import os
import torch
import numpy as np
from transformers import AutoModel,AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation import GenerationConfig
from argparse import ArgumentParser
import json
from tqdm import tqdm
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
from PIL import Image
import torchvision.transforms as T
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType,
    get_default_template_type, inference_stream
)
from swift.utils import seed_everything
import torch
#from transformers import Qwen2VLForConditionalGeneration
device = 'cuda'

model_path_map = {
    "Qwen2-VL-2B-Instruct": "/home/gpuall/ifs_data/pre_llms/Qwen2/Qwen2-VL-2B-Instruct",
    "Qwen2-VL-7B-Instruct": "/home/gpuall/ifs_data/pre_llms/Qwen2/Qwen2-VL-7B-Instruct",
    "MiniCPM-V-2_5": "/home/gpuall/ifs_data/pre_llms/MiniCPM-Llama3-V-2_5",
    "MiniCPM-V-2.6": "/home/gpuall/ifs_data/pre_llms/MiniCPM-V-2_6",
    "InternVL-chat-1.5": "/home/gpuall/ifs_data/pre_llms/InternVL-Chat-V1-5",
    "InternVL-2.0": "/home/gpuall/ifs_data/pre_llms/InternVL2-8B",
    "Yi-VL-6B":"/home/gpuall/ifs_data/pre_llms/Yi-VL-6B/Yi-VL-6B",
    "glm-4v-9b":"/home/gpuall/ifs_data/pre_llms/ZhipuAI/glm-4v-9b",

}

model_name_list = list(model_path_map.keys())
print('允许输入的模型名称:')
print(model_name_list)

tasks_list = [
    "文本翻译",
    "OCR",
    "图片理解",
    "图文一致性判断"
]

model_name_list = list(model_path_map.keys())
print('允许输入的模型名称:')
print(model_name_list)
prompt_type_list = ['0_shot', '5_shot', 'chain_of_thought']

def read_json(path):
    with open(path, 'r',encoding='utf-8') as file:
       data = json.load(file)
    return data

def save_as_json(path,responses,data):
    with open(path,'w', encoding='utf-8') as fp:
        for i,j in zip(data,responses):
            data={'instruction':i['instruction'],'text':i['input'],'raw':i['output'],'predicate':j}
            jsonstr=json.dumps(data, ensure_ascii=False)
            fp.writelines(jsonstr + '\n')

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, default='',choices=model_name_list,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--task_name", type=str, default='',choices=tasks_list,
                        help="task name, default to %(default)r")
    parser.add_argument("--prompt_type", type=str, default='',choices=prompt_type_list,
                        help="prompt engniering, default to %(default)r")
    parser.add_argument("--data_path", type=str, default='',
                        help="used data path")
    parser.add_argument("--sample", type=int, default=None,
                        help="num for elements selected in list")
    args = parser.parse_args()
    return args

args = get_args()
print('当前执行的任务是:'+args.prompt_type+args.task_name)
print('当前使用的模型是:'+args.model_name)
class ChatModelProcess:
    def __init__(self, model_name,task_name,prompt_type,data_path,sample):
        self.models = model_name
        self.task_name = task_name
        self.prompt_type = prompt_type
        self.data_path = data_path
        self.check_point = model_path_map.get(model_name)
        self.data_all=read_json(path=data_path)
        if sample is None:
            self.data=self.data_all
        else:
            self.data=self.data_all[:args.sample]

    def Qwen2_vl_7b_instruct(self):
        print('开始调用Qwen2-VL系列的模型')
        model_path = self.check_point
        data = self.data

        model_type = ModelType.qwen2_vl_7b_instruct
        model_id_or_path = model_path
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)

        responses = []
        data = self.data
        for item in tqdm(data):
            prompt = item['instruction'] + item['input']
            image_paths = item['images']

            # 确认 image_paths 是一个列表
            if not isinstance(image_paths, list):
                raise ValueError("image_paths should be a list of image paths.")

            if len(image_paths) == 0:
                raise ValueError("No image paths provided.")

            # 加载图像
            images = [Image.open(path).convert('RGB') for path in image_paths if os.path.exists(path)]

            query = '<image>' + prompt  # 拼接成最终的提示符
            gen = inference_stream(model, template, query, images=images)

            print_idx = 0
            print(f'query: {query}\nresponse: ', end='')

            # 存储响应
            response_text = ''

            # 实时打印生成的响应
            for response, history in gen:
                delta = response[print_idx:]
                print(delta, end='', flush=True)
                response_text += delta
                print_idx = len(response)

            print()  # 新行以便于下一条记录的打印
            responses.append(response_text)  # 将生成的响应追加到列表中

        return responses

    def Qwen2_vl_2b_instruct(self):
        print('开始调用Qwen2-VL系列的模型')
        model_path = self.check_point
        data = self.data

        model_type = ModelType.qwen2_vl_7b_instruct
        model_id_or_path = model_path
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)

        responses = []
        data = self.data
        for item in tqdm(data):
            prompt = item['instruction'] + item['input']
            image_paths = item['images']

            # 确认 image_paths 是一个列表
            if not isinstance(image_paths, list):
                raise ValueError("image_paths should be a list of image paths.")

            if len(image_paths) == 0:
                raise ValueError("No image paths provided.")

            # 加载图像
            images = [Image.open(path).convert('RGB') for path in image_paths if os.path.exists(path)]

            query = '<image>' + prompt  # 拼接成最终的提示符
            gen = inference_stream(model, template, query, images=images)

            print_idx = 0
            print(f'query: {query}\nresponse: ', end='')

            # 存储响应
            response_text = ''

            # 实时打印生成的响应
            for response, history in gen:
                delta = response[print_idx:]
                print(delta, end='', flush=True)
                response_text += delta
                print_idx = len(response)

            print()  # 新行以便于下一条记录的打印
            responses.append(response_text)  # 将生成的响应追加到列表中

        return responses
    def MinCPM(self):
        print('开始调用 MinCPM 系列的模型')
        model_path = self.check_point
        data = self.data

        model = AutoModelForCausalLM.from_pretrained(
            model_path,  # 使用本地路径加载模型
            torch_dtype=torch.float16,  # 确保与设备兼容
            device_map={"": "cuda:0"},  # 手动指定设备
            trust_remote_code=True  # 允许执行自定义代码
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True  # 允许执行自定义代码
        )  # 使用本地路径加载 tokenizer

        data=self.data
        # responses=[]
        texts=[]
        responses=[]
        for i in tqdm(data):
            prompt = i['instruction']+i['input']
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            responses.append(response)


    def InternVL_chat_v1_5(self):
        print('开始调用 InternVL 系列的模型')
        model_path = self.check_point
        data = self.data

        model_type = "internvl-chat-v1_5"
        model_id_or_path = model_path
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)
        data=self.data
        # responses=[]
        texts=[]
        responses=[]
        for i in tqdm(data):
            prompt = i['instruction']+i['input']
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            generated_ids = model.generate(
                model_inputs.input_ids,
                max_new_tokens=1024
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(response)
            responses.append(response)

    def InternVL_2(self):
        print('开始调用 InternVL 系列的模型')
        model_path = self.check_point
        data = self.data

        model_type = "internvl2-8b"
        model_id_or_path = model_path
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)

        responses = []
        data = self.data
        for item in tqdm(data):
            prompt = item['instruction'] + item['input']
            image_paths = item['images']

            # 确认 image_paths 是一个列表
            if not isinstance(image_paths, list):
                raise ValueError("image_paths should be a list of image paths.")

            if len(image_paths) == 0:
                raise ValueError("No image paths provided.")

            # 加载图像
            images = [Image.open(path).convert('RGB') for path in image_paths if os.path.exists(path)]

            query = '<image>' + prompt  # 拼接成最终的提示符
            gen = inference_stream(model, template, query, images=images)

            print_idx = 0
            print(f'query: {query}\nresponse: ', end='')

            # 存储响应
            response_text = ''

            # 实时打印生成的响应
            for response, history in gen:
                delta = response[print_idx:]
                print(delta, end='', flush=True)
                response_text += delta
                print_idx = len(response)

            print()  # 新行以便于下一条记录的打印
            responses.append(response_text)  # 将生成的响应追加到列表中

        return responses

    def Yi(self):
        print('开始调用 Yi-VL系列的模型')
        model_path = self.check_point
        data = self.data

        model_type = ModelType.yi_vl_6b_chat
        model_id_or_path = model_path
        template_type = get_default_template_type(model_type)
        print(f'template_type: {template_type}')

        model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16, model_id_or_path=model_id_or_path,
                                       model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)

        responses = []
        data = self.data
        for item in tqdm(data):
            prompt = item['instruction'] + item['input']
            image_paths = item['images']

            # 确认 image_paths 是一个列表
            if not isinstance(image_paths, list):
                raise ValueError("image_paths should be a list of image paths.")

            if len(image_paths) == 0:
                raise ValueError("No image paths provided.")

            # 加载图像
            images = [Image.open(path).convert('RGB') for path in image_paths if os.path.exists(path)]

            query = '<image>' + prompt  # 拼接成最终的提示符
            gen = inference_stream(model, template, query, images=images)

            print_idx = 0
            print(f'query: {query}\nresponse: ', end='')

            # 存储响应
            response_text = ''

            # 实时打印生成的响应
            for response, history in gen:
                delta = response[print_idx:]
                print(delta, end='', flush=True)
                response_text += delta
                print_idx = len(response)

            print()  # 新行以便于下一条记录的打印
            responses.append(response_text)  # 将生成的响应追加到列表中

        return responses

    def GLM(self):
        print('开始调用"GLM-4V-9B"模型')
        model_path = self.check_point
        data = self.data

        model = AutoModelForCausalLM.from_pretrained(
        model_path,  # 使用本地路径加载模型
        torch_dtype=torch.float16,  # 确保与设备兼容
        device_map={"": "cuda:0"},  # 手动指定设备
        trust_remote_code=True  # 允许执行自定义代码
        )

        tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True  # 允许执行自定义代码
        )  # 使用本地路径加载 tokenizer

        responses = []
        data = self.data
        for item in tqdm(data):
            prompt = item['instruction'] + item['input']
            image_paths = item['images']

            if not isinstance(image_paths, list):
                raise ValueError("image_paths should be a list of image paths.")

            if len(image_paths) == 0:
                raise ValueError("No image paths provided.")

            # 处理第一个图像路径
            image_path = image_paths[0]
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"The file {image_path} does not exist.")

            # 打开并转换图像
            image = Image.open(image_path).convert('RGB')

            # 构建查询字符串
            query = prompt

            # 应用聊天模板
            inputs = tokenizer.apply_chat_template(
                [{"role": "user", "image": image, "content": query}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True  # chat mode
            )

            # 将输入张量移动到正确的设备
            inputs = {k: v.to('cuda:0') for k, v in inputs.items() if isinstance(v, torch.Tensor)}

            gen_kwargs = {
                "max_length": 2500,
                "do_sample": True,
                "top_k": 1
            }

            # 执行推理并生成输出
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 将结果追加到响应列表中
            responses.append(decoded_output)

        return responses


    def call_method_by_model_name(self):
        # 映射 model_name 到类方法
        model_to_method = {
            "Qwen2-VL-2B-Instruct": self.Qwen2_vl_2b_instruct,
            "Qwen2-VL-7B-Instruct": self.Qwen2_vl_7b_instruct,
            "MiniCPM-V-2_5": self.MinCPM,
            "MiniCPM-V-2.6": self.MinCPM,
            "InternVL-chat-1.5": self.InternVL_chat_v1_5,
            "InternVL-2.0": self.InternVL_2,
            "Yi-VL-6B": self.Yi,
            "GLM-4V-9B":self.GLM,
        }

        # 获取对应的方法并调用
        method = model_to_method.get(self.models)
        if method:
            return method()
        else:
            raise ValueError(f"No method found for model_name: {self.models}")

 ##测试效果
chat_model_process = ChatModelProcess(
   model_name='InternVL-chat-1.5',
        task_name='文本翻译',
     prompt_type='0_shot',
     data_path='/home/gpuall/ifs_data/data/ancient_data/gujitupian/pingce/data/wenben_data/24史_sample_100.json',
    sample=args.sample,
)
# # 根据模型名称调用对应的方法
response = chat_model_process.call_method_by_model_name()
if response is None:
    print("Error: Response is None.")
else:
    print("Responses collected successfully.")
    # 继续保存结果到文件
    save_as_json(path='/home/gpuall/zdm/LLM/pingce/jieguo/fanyijieguo/InternVL-chat-1.5_24史_文本翻译_0_shot_100.jsonl', responses=response, data=data)


def create_directory(directory_path):
    try:
        # 检查文件夹是否存在
        if not os.path.exists(directory_path):
            # 创建文件夹
            os.makedirs(directory_path)
            print(f"文件夹 '{directory_path}' 创建成功。")
        else:
            print(f"文件夹 '{directory_path}' 已经存在。")
    except Exception as e:
        print(f"创建文件夹时出错: {e}")

chat_model_process = ChatModelProcess(
    model_name=args.model_name,
    task_name=args.task_name,
    prompt_type=args.prompt_type,
    data_path=args.data_path,
    sample=args.sample
)
# 根据模型名称调用对应的方法
response = chat_model_process.call_method_by_model_name()
data=read_json(path=args.data_path)
dir_path='result/'+args.task_name+'_'+args.prompt_type
create_directory(directory_path=dir_path)
save_as_json(path=dir_path+'/'+args.model_name+'_'+args.task_name+'_'+args.prompt_type+'.json',responses=response,data=data)
