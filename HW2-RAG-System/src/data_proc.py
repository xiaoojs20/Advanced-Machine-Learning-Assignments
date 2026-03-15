
import bs4
from pdfminer.high_level import extract_text
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.document_loaders import BSHTMLLoader
import re
import os
import pdfplumber
import logging
import json
from tqdm import tqdm

with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)
    

def get_data_path(config):
    """
    get all data path from config file
    """
    base_path = config['data']['path']
    content = config['data']['content']
    
    all_paths = []
    for item in content:
        path = os.path.join(base_path, item)
        all_paths.append(path)
    
    return all_paths

def load_files(path):
    """
    load all pdf/html files in the directory
    """
    loaders = []
    for file in os.listdir(path):
        file_type = file.split('.')[-1]
        # if file.lower().endswith('.pdf'):
        if file_type == 'pdf':
            print(f"Processing: {file}")
            file_path = os.path.join(path, file)
            loader = PyMuPDFLoader(file_path)
            loaders.append(loader)
        elif file_type == 'html':
            print(f"Processing: {file}")
            file_path = os.path.join(path, file)
            loader = BSHTMLLoader(file_path)
            loaders.append(loader)
            
    return loaders

def get_url(config):
    """
    get all url from config file
    """
    return config['data']['url']

def load_url_contents(urls):
    """
    load all url contents, 无法访问外网
    """
    loaders = []
    # Only keep post title, headers, and content from the full HTML.
    for url in urls:
        bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
        loader = WebBaseLoader(
            web_paths=("https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",),
            bs_kwargs={"parse_only": bs4_strainer},
        )
        docs = loader.load()
        assert len(docs) == 1
        # print(f"Total characters: {len(docs[0].page_content)}")      
        loaders.append(loader)
    return loaders





# 设置日志记录
logging.basicConfig(
    filename='pdf_processing.log',
    level=logging.ERROR,
    format='%(asctime)s:%(levelname)s:%(message)s'
)


# ee_path = "/flash2/aml/xiaojs24/AML_2/database/EE"
pumpkin_book_path = "/flash2/aml/xiaojs24/AML_2/database/pumpkin"
PATH = pumpkin_book_path

## convert all pdf files in the directory to txt files
def extract_text_pdfplumber(pdf_path):
    """
    使用pdfplumber提取PDF文本
    """
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                else:
                    logging.warning(f"No text found on page {page_num} in {os.path.basename(pdf_path)}")
    except Exception as e:
        logging.error(f"Error extracting text from {os.path.basename(pdf_path)}: {e}")
        raise e  # 重新抛出异常以便外层处理
    return text


def convert_pdfs_to_txt(path):
    """
    将指定目录下的所有PDF文件转换为TXT文件
    """
    for filename in os.listdir(path):
        if filename.lower().endswith(".pdf"):
            print(f"Processing: {filename}")
            pdf_path = os.path.join(path, filename)
            
            try:
                # 使用pdfplumber提取文本
                text = extract_text_pdfplumber(pdf_path)
                
                # 构造txt文件的文件名（pdf同名）
                txt_filename = os.path.splitext(filename)[0] + ".txt"
                txt_path = os.path.join(path, txt_filename)
                
                # 将提取的文本写入txt文件
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                
                print(f"Converted: {filename} -> {txt_filename}")
            
            except Exception as e:
                print(f"Failed to process {filename}: {e}")
                logging.error(f"Failed to process {filename}: {e}")
                

def main():
    # Load data
    loaders = []
    paths = get_data_path(config)
    # urls = get_url(config)
    
    for path in paths:
        loader = load_files(path)
        loaders.extend(loader)
        
    # for url in urls:
    #     loader = load_url_contents(url)
    #     loaders.extend(loader)
    
    print(f"paths len = {len(paths)}")
    print(f"loaders len = {len(loaders)}")
    
    # Clean data
    print("Before cleaning:")
    pages = loaders[0].load()
    page = pages[0]
    print(f"每一个元素的类型：{type(page)}.", 
        f"该文档的描述性数据：{page.metadata}", 
        f"查看该文档的内容:\n{page.page_content}", 
        sep="\n------\n")
    
    pattern = re.compile(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]', re.DOTALL)
    # page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), page.page_content)
    # page.page_content = page.page_content.replace('•', '')
    # print(page.page_content)
    
    
    with open('data.txt', 'w', encoding='utf-8') as f:
        for loader in loaders:
            for page in loader.load():
                page.page_content = re.sub(pattern, lambda match: match.group(0).replace('\n', ''), page.page_content)
                page.page_content = page.page_content.replace('•', '')
                print(page.page_content)
                f.write(page.page_content + '\n')


    # print("After cleaning:")
    # pages = loaders[4].load()
    # page = pages[3]
    # print(f"每一个元素的类型：{type(page)}.", 
    #     f"该文档的描述性数据：{page.metadata}", 
    #     f"查看该文档的内容:\n{page.page_content}", 
    #     sep="\n------\n")
    
    # texts = []
    # for loader in loaders: texts.extend(loader.load())
    
    
if __name__ == "__main__":
    main()
    

        
    
    

    

    





    
    
    # convert_pdfs_to_txt(PATH)
    
    
    
    
# for filename in os.listdir(PATH):
#     if filename.lower().endswith(".pdf"):
#         print(f"Processing: {filename}")
#         pdf_path = os.path.join(PATH, filename)
        
#         # 使用pdfminer提取文本
#         text = extract_text(pdf_path)
        
#         # 构造txt文件的文件名（pdf同名）
#         txt_filename = os.path.splitext(filename)[0] + ".txt"
#         txt_path = os.path.join(PATH, txt_filename)
        
#         # 将提取的文本写入txt文件
#         with open(txt_path, "w", encoding="utf-8") as f:
#             f.write(text)

#         print(f"Converted: {filename} -> {txt_filename}")


# ## clean all txt files in the directory

# # 定义允许保留的字符集合：
# # - 汉字：[\u4e00-\u9fa5]
# # - 英文字母：a-zA-Z
# # - 数字：0-9
# # - 常用标点和符号

# allowed_chars = '[-\u4e00-\u9fa5a-zA-Z0-9 .,;:?!%(){}<>《》“”‘’"\'—_+*=/、，。；：“”‘’（）]+'

# txt_files = [f for f in os.listdir(PATH) if f.lower().endswith('.txt')]
# txt_files.sort()

# all_texts = []

# for txt_filename in txt_files:
#     txt_path = os.path.join(PATH, txt_filename)
    
#     with open(txt_path, "r", encoding="utf-8") as f:
#         content = f.read()
        
#     # 1. 去除控制字符和不可见字符
#     #   匹配所有Unicode控制字符并删掉
#     content = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', content)
    
#     # 2. 使用 allowed_chars 过滤不需要的字符
#     #   findall 保留符合规则的字符，然后再 join 回去
#     filtered_parts = re.findall(allowed_chars, content)
#     content = "".join(filtered_parts)
    
#     # 3. 将连续空白合并为单一空格，并strip去除首尾空白
#     content = re.sub(r'\s+', ' ', content).strip()
    
#     # 去除特定的页眉页脚模式，例如如果原PDF每页都有类似 "第 X 页"
#     content = re.sub(r'第\s*\d+\s*页', '', content)
    
#     # 将清洗后的文本加入列表
#     all_texts.append(content)

# # 将所有文件的清洗结果合并成一个大的txt文件
# merged_text = "\n\n".join(all_texts)
# merged_path = os.path.join(PATH, "pumpkin_book.txt")
# with open(merged_path, "w", encoding="utf-8") as f:
#     f.write(merged_text)

# print(f"All texts have been merged into {merged_path}")