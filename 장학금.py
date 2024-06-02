import json
import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch
import pickle
from pingpong import PingPong
from pingpong.gradio import GradioAlpacaChatPPManager
from tqdm import tqdm
import pandas as pd


STYLE = """
.custom-btn {
    border: none !important;
    background: none !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}
.custom-btn:hover {
    background: rgb(243 244 246) !important;
}

.custom-btn-highlight {
    border: none !important;
    background: rgb(243 244 246) !important;
    box-shadow: none !important;
    display: block !important;
    text-align: left !important;
}

#prompt-txt > label > span {
    display: none !important;
}
#prompt-txt > label > textarea {
    border: transparent;
    box-shadow: none;
}
#chatbot {
    height: 800px; 
    overflow: auto;
    box-shadow: none !important;
    border: none !important;
}
#chatbot > .wrap {
    max-height: 780px;
}
#chatbot + div {
  border-radius: 35px !important;
  width: 80% !important;
  margin: auto !important;  
}

#left-pane {
    background-color: #f9fafb;
    border-radius: 15px;
    padding: 10px;
}

#left-top {
    padding-left: 10px;
    padding-right: 10px;
    text-align: center;
    font-weight: bold;
    font-size: large;    
}

#chat-history-accordion {
    background: transparent;
    border: 0.8px !important;  
}

#right-pane {
  margin-left: 20px;
  margin-right: 70px;
}

#initial-popup {
    z-index: 100;
    position: absolute;
    width: 50%;
    top: 50%;
    height: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    border-radius: 35px;
    padding: 15px;
}

#initial-popup-title {
    text-align: center;
    font-size: 18px;
    font-weight: bold;    
}

#initial-popup-left-pane {
    min-width: 150px !important;
}

#initial-popup-right-pane {
    text-align: right;
}

.example-btn {
    padding-top: 20px !important;
    padding-bottom: 20px !important;
    padding-left: 5px !important;
    padding-right: 5px !important;
    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;
}

.example-btn:hover {
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}

#example-title {
  margin-bottom: 15px;
}

#aux-btns-popup {
    z-index: 200;
    position: absolute !important;
    bottom: 75px !important;
    right: 15px !important;
}

#aux-btns-popup > div {
    flex-wrap: nowrap;
    width: auto;
    margin: auto;  
}

.aux-btn {
    height: 30px !important;
    flex-wrap: initial !important;
    flex: none !important;
    min-width: min(100px,100%) !important;
    font-weight: unset !important;
    font-size: 10pt !important;

    background: linear-gradient(to bottom right, #f7faff, #ffffff) !important;
    box-shadow: none !important;
    border-radius: 20px !important;    
}

.aux-btn:hover {
    box-shadow: 0.3px 0.3px 0.3px gray !important;
}
"""

get_local_storage = """
function() {
  globalThis.setStorage = (key, value)=>{
    localStorage.setItem(key, JSON.stringify(value));
  }
  globalThis.getStorage = (key, value)=>{
    return JSON.parse(localStorage.getItem(key));
  }

  var local_data = getStorage('local_data');
  var history = [];

  if(local_data) {
    local_data[0].pingpongs.forEach(element =>{ 
      history.push([element.ping, element.pong]);
    });
  }
  else {
    local_data = [];
    for (let step = 0; step < 10; step++) {
      local_data.push({'ctx': '', 'pingpongs':[]});
    }
    setStorage('local_data', local_data);
  }

  if(history.length == 0) {
    document.querySelector("#initial-popup").classList.remove('hide');
  }

  return [history, local_data];
}
"""

update_left_btns_state = """
(v)=>{
  document.querySelector('.custom-btn-highlight').classList.add('custom-btn');
  document.querySelector('.custom-btn-highlight').classList.remove('custom-btn-highlight');

  const elements = document.querySelectorAll(".custom-btn");

  for(var i=0; i < elements.length; i++) {
    const element = elements[i];
    if(element.textContent == v) {
      console.log(v);
      element.classList.add('custom-btn-highlight');
      element.classList.remove('custom-btn');
      break;
    }
  }
}"""

channels = [
    "1st Channel",
    "2nd Channel",
    "3rd Channel",
    "4th Channel",
    "5th Channel",
    "6th Channel",
    "7th Channel",
    "8th Channel",
    "9th Channel",
    "10th Channel",
]
channel_btns = []

examples = [
    "hello world",
    "what's up?",
    "this is GradioChat"
]
ex_btns = []

class ScholarshipFiltering:
    def __init__(self, model: AutoModel,
                 tokenizer: AutoTokenizer,
                 filtering_cols,
                 device,
                 scholarship_feature=None,
                 scholarship_df=None):

        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.filtering_cols = filtering_cols
        self.scholarship_feature = scholarship_feature
        self.scholarship_df = scholarship_df

        if self.scholarship_feature is None:
            assert self.scholarship_df is not None, "There is no dataframe data for extraction"

            self.scholarship_feature = {f"{company}_{name}": {col: None for col in filtering_cols} for company, name in zip(self.scholarship_df["운영기관명"], self.scholarship_df["상품명"])}
            self.init_feature()

        self.history = []
        self.find_list = []

    def init_feature(self):
        with torch.no_grad():
            for idx, (company, name) in tqdm(enumerate(zip(self.scholarship_df["운영기관명"], self.scholarship_df["상품명"])), total=self.scholarship_df.shape[0]):
                for col in self.filtering_cols:
                    text = self.df[col][idx]
                    text_input = self.tokenizer(text, max_length=512, padding="max_length", return_tensors="pt").to(self.device)

                    out = self.model(**text_input)
                    self.scholarship_feature[f"{company}_{name}"][col] = out.pooler_output.cpu()

    def storing_sim_with_cols(self, input_text: str, col):
        self.history.append(input_text)
        input_text = self.tokenizer(input_text, max_length=512, padding="max_length", return_tensors="pt").to(self.device)
        feature = self.model(**input_text).pooler_output.cpu()

        f_list = torch.stack([self.scholarship_feature[name][col] for name in self.scholarship_feature.keys()]).squeeze(1)
        sim = torch.cosine_similarity(feature, f_list)
        sim = torch.sort(sim, descending=True)

        self.find_list.append(sim.indices[sim.values > 0.5].tolist())
    
    def get_scholarship_info(self, indices):
        result_df = self.scholarship_df.iloc[indices]
        result_strings = []
        for idx, row in result_df.iterrows():
            info = f"""
            장학금: {row['상품명']}
            학자금유형: {row['학자금유형구분']}
            운영기관: {row['운영기관명']}
            대학구분: {row['대학구분']}
            학년구분: {row['학년구분']}
            학과구분: {row['학과구분']}
            성적기준: {row['성적기준 상세내용']}
            소득기준: {row['소득기준 상세내용']}
            특정자격: {row['특정자격 상세내용']}
            지역거주여부: {row['지역거주여부 상세내용']}
            선발방법: {row['선발방법 상세내용']}
            선발인원: {row['선발인원 상세내용']}
            자격제한: {row['자격제한 상세내용']}
            추천필요여부: {row['추천필요여부 상세내용']}
            제출서류: {row['제출서류 상세내용']}
            홈페이지: {row['홈페이지 주소']}
            """
            result_strings.append(info.strip())
        return "\n\n".join(result_strings)

    def filtering(self):
        common_indices = set(self.find_list[0])
        for lst in self.find_list[1:]:
            common_indices &= set(lst)

        common_indices = list(common_indices)

        return common_indices[:2]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "BM-K/KoSimCSE-roberta"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)

# feature 파일 로드
with open('feature_dict.pkl', 'rb') as f:
    scholarship_feature = pickle.load(f)

filtering_cols = ["대학구분", "학년구분", "학과구분"]

# 로드된 feature 파일로 ScholarshipFiltering 초기화
scholarship_df = pd.read_csv("대학생.csv", encoding="cp949")

filtering = ScholarshipFiltering(model, tokenizer, filtering_cols, device, scholarship_feature=scholarship_feature, scholarship_df=scholarship_df)
# 대화 흐름
conversation_flow = [
    "몇년제 대학 다니나요?",
    "몇 학기인가요?",
    "어떤 계열에 재학 중인가요?"
]
conversation_states = {}  # 각 채널의 상태를 추적하기 위해 사용

def add_pingpong(idx, ld, ping):
    res = [GradioAlpacaChatPPManager.from_json(json.dumps(ppm)) for ppm in ld]
    ppm = res[idx]

    channel = f"Channel_{idx}"
    if channel not in conversation_states:
        conversation_states[channel] = {"step": 0, "answers": []}

    current_state = conversation_states[channel]
    
    # 사용자의 인사말을 무시하고 질문을 시작하도록 로직 수정
    greetings = ["장학금 추천해줘", "안녕하세요", "hi", "hello", "장학금"]  # 예상되는 인사말 목록
    if ping.lower() in greetings and current_state["step"] == 0:
        # 인사말이 들어올 경우, 첫 번째 질문으로 응답
        response = conversation_flow[current_state["step"]]
    else:
        # 인사말이 아닌 경우, 정상적으로 대화를 진행
        current_state["answers"].append(ping)
        current_state["step"] += 1
        if current_state["step"] < len(conversation_flow):
            response = conversation_flow[current_state["step"]]
        else:
            # 모든 질문에 답변이 완료되면 필터링 수행
            for i, answer in enumerate(current_state["answers"]):
                filtering.storing_sim_with_cols(answer, filtering_cols[i])
            filtered_indices = filtering.filtering()
            scholarship_info = filtering.get_scholarship_info(filtered_indices)
            response = f"Filtered results: {scholarship_info}"
            # 다음 질문 라운드를 위해 대화 상태 초기화
            current_state["step"] = 0
            current_state["answers"] = []

    ppm.add_pingpong(PingPong(ping, response))
    return "", ppm.build_uis(), str(res)

def channel_num(btn_title):
    choice = 0
    for idx, channel in enumerate(channels):
        if channel == btn_title:
            choice = idx
    return choice

def set_chatbot(btn, ld):
    choice = channel_num(btn)
    res = [GradioAlpacaChatPPManager.from_json(json.dumps(ppm_str)) for ppm_str in ld]
    empty = len(res[choice].pingpongs) == 0
    return (
        res[choice].build_uis(),
        choice,
        gr.update(visible=empty)
    )

def set_example(btn):
    return btn, gr.update(visible=False)

def set_popup_visibility(ld, example_block):
    return example_block

with gr.Blocks(css=STYLE, elem_id='container-col') as block:
    idx = gr.State(0)
    local_data = gr.JSON({}, visible=False)

    with gr.Row():
        with gr.Column(scale=1, min_width=180):
            gr.Markdown("GradioChat", elem_id="left-top")

            with gr.Column(elem_id="left-pane"):
                with gr.Accordion("Histories", elem_id="chat-history-accordion"):
                    channel_btns.append(gr.Button(channels[0], elem_classes=["custom-btn-highlight"]))

                    for channel in channels[1:]:
                        channel_btns.append(gr.Button(channel, elem_classes=["custom-btn"]))

        with gr.Column(scale=8, elem_id="right-pane"):
            with gr.Column(elem_id="initial-popup", visible=False) as example_block:
                with gr.Row():
                    with gr.Column(elem_id="initial-popup-left-pane"):
                        gr.Markdown("GradioChat", elem_id="initial-popup-title")
                        gr.Markdown("Making the community's best AI chat models available to everyone.")
                    with gr.Column(elem_id="initial-popup-right-pane"):
                        gr.Markdown("Chat UI is now open sourced on Hugging Face Hub")
                        gr.Markdown("check out the [↗ repository](https://huggingface.co/spaces/chansung/test-multi-conv)")

                with gr.Column(scale=1):
                    gr.Markdown("Examples")
                    with gr.Row() as text_block:
                        for example in examples:
                            ex_btns.append(gr.Button(example, elem_classes=["example-btn"]))

            with gr.Column(elem_id="aux-btns-popup", visible=True):
                with gr.Row():
                    stop = gr.Button("Stop", elem_classes=["aux-btn"])
                    regenerate = gr.Button("Regenerate", elem_classes=["aux-btn"])
                    clean = gr.Button("Clean", elem_classes=["aux-btn"])

            chatbot = gr.Chatbot(elem_id='chatbot')
            instruction_txtbox = gr.Textbox(
                placeholder="Ask anything", label="",
                elem_id="prompt-txt"
            )

    for btn in channel_btns:
        btn.click(
            set_chatbot,
            [btn, local_data],
            [chatbot, idx, example_block]
        ).then(
            None, btn, None,
            js=update_left_btns_state
        )

    for btn in ex_btns:
        btn.click(
            set_example,
            [btn],
            [instruction_txtbox, example_block]
        )

    instruction_txtbox.submit(
        lambda: gr.update(visible=False),
        None,
        example_block
    ).then(
        add_pingpong,
        [idx, local_data, instruction_txtbox],
        [instruction_txtbox, chatbot, local_data]
    ).then(
        None, local_data, None,
        js="(v)=>{ setStorage('local_data',v) }"
    )

    block.load(
        None,
        inputs=None,
        outputs=[chatbot, local_data],
        js=get_local_storage,
    )

block.queue().launch(debug=True, share=True)  # 공개 링크를 만들려면 share=True 설정
