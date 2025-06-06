# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 13:43:16 2025

@author: Lily
"""

import re
import jieba
import pandas as pd
from tqdm import tqdm
import os
import sys

# 配置jieba分词器
jieba.setLogLevel('WARN')  # 减少日志输出

class SinaDataPreprocessor:
    def __init__(self, stopwords_path=None):
        """
        初始化新浪数据预处理器
        :param stopwords_path: 停用词表文件路径
        """
        # 加载停用词表
        if stopwords_path and os.path.exists(stopwords_path):
            self.stopwords = self.load_stopwords(stopwords_path)
        else:
            # 如果停用词表文件不存在，使用默认停用词
            self.stopwords = self.get_default_stopwords()
        
        # 添加新浪数据集特有的停用词
        sina_stopwords = {'新浪', '新闻', '报道', '讯', '消息', '网', '客户端', '快讯', '记者', '日电', '日报'}
        self.stopwords.update(sina_stopwords)
    
    @staticmethod
    def load_stopwords(file_path):
        """加载停用词表"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return set([line.strip() for line in f])
    
    @staticmethod
    def get_default_stopwords():
        """获取默认停用词表"""
        return {
            '的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '在', '中', '为', '上', '下', '个', '之', '或',
            '有', '更', '好', '大', '小', '多', '少', '很', '只', '不', '也', '又', '还', '要', '没', '没', '没有', '我们', '他们',
            '你们', '它', '它们', '这个', '那个', '这些', '那些', '什么', '怎么', '为什么', '如何', '何时', '何地', '谁', '哪', '吧',
            '啊', '呀', '呢', '吗', '啦', '哦', '唉', '嗯', '呃', '啊呀', '嘛', '罢了', '着呢', '得', '地', '着', '过', '来着'
        }
    
    def clean_text(self, text):
        """清理文本：移除特殊标签、符号和多余空格"""
        if not isinstance(text, str):
            return ""
        
        # 移除[新浪新闻]类标签
        text = re.sub(r'\[.*?\]|\【.*?\】', '', text)
        
        # 移除特殊符号和标点
        text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
        
        # 合并多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_text(self, text):
        """分词并去除停用词"""
        cleaned_text = self.clean_text(text)
        
        # 使用jieba进行中文分词
        words = jieba.lcut(cleaned_text)
        
        # 去除停用词并过滤单字
        filtered_words = [
            word for word in words 
            if (word not in self.stopwords) and (len(word) > 1) and (not word.isdigit())
        ]
        
        return filtered_words
    
    def preprocess_data(self, input_file, output_file='data.csv'):
        """
        预处理整个数据集
        :param input_file: 输入数据集文件路径
        :param output_file: 输出文件路径
        """
        # 方法1：尝试使用pandas读取
        print("尝试方法1：使用pandas读取CSV...")
        try:
            # 尝试使用不同编码
            for encoding in ['utf-8', 'gbk', 'latin1']:
                try:
                    df = pd.read_csv(input_file, encoding=encoding, engine='python')
                    print(f"成功使用编码: {encoding}")
                    print(f"数据集已加载，共 {len(df)} 条记录")
                    print(f"数据列: {list(df.columns)}")
                    return self.process_dataframe(df, output_file)
                except (UnicodeDecodeError, pd.errors.ParserError):
                    continue
        except Exception as e:
            print(f"方法1失败: {str(e)}")
        
        # 方法2：尝试手动读取文件
        print("\n尝试方法2：手动读取文件...")
        try:
            # 尝试不同编码
            for encoding in ['utf-8', 'gbk', 'latin1']:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    
                    # 尝试确定列数
                    header = lines[0].strip().split(',')
                    num_columns = len(header)
                    
                    # 创建数据列表
                    data = []
                    for line in lines[1:]:
                        parts = line.strip().split(',')
                        if len(parts) == num_columns:
                            data.append(parts)
                        elif len(parts) > num_columns:
                            # 处理可能包含逗号的文本字段
                            merged = parts[:num_columns-1] + [','.join(parts[num_columns-1:])]
                            if len(merged) == num_columns:
                                data.append(merged)
                    
                    # 创建DataFrame
                    df = pd.DataFrame(data, columns=header)
                    print(f"成功使用编码: {encoding}")
                    print(f"数据集已加载，共 {len(df)} 条记录")
                    print(f"数据列: {list(df.columns)}")
                    return self.process_dataframe(df, output_file)
                except (UnicodeDecodeError, Exception) as e:
                    print(f"编码 {encoding} 失败: {str(e)}")
        except Exception as e:
            print(f"方法2失败: {str(e)}")
        
        # 方法3：尝试读取为纯文本
        print("\n尝试方法3：读取为纯文本...")
        try:
            # 尝试不同编码
            for encoding in ['utf-8', 'gbk', 'latin1']:
                try:
                    with open(input_file, 'r', encoding=encoding) as f:
                        text = f.read()
                    
                    # 直接创建DataFrame，只包含一个文本列
                    df = pd.DataFrame({'text': text.split('\n')})
                    print(f"成功使用编码: {encoding}")
                    print(f"数据集已加载，共 {len(df)} 条记录")
                    print("警告：无法解析列结构，将所有内容视为单一文本列")
                    return self.process_dataframe(df, output_file)
                except (UnicodeDecodeError, Exception) as e:
                    print(f"编码 {encoding} 失败: {str(e)}")
        except Exception as e:
            print(f"方法3失败: {str(e)}")
        
        # 所有方法都失败
        raise ValueError("所有读取方法均失败，无法处理文件")
    
    def process_dataframe(self, df, output_file):
        """处理DataFrame并保存结果"""
        # 查找可能的文本列
        text_columns = [col for col in df.columns if 'content' in col.lower() or 'text' in col.lower() or 'news' in col.lower()]
        
        if not text_columns:
            # 如果没有找到标准列名，尝试使用第一列
            text_columns = [df.columns[0]]
            print(f"警告: 未找到标准文本列，将使用第一列 '{text_columns[0]}' 作为文本内容")
        else:
            text_column = text_columns[0]
            print(f"使用列 '{text_column}' 作为文本内容列")
        
        # 添加进度条
        tqdm.pandas(desc="文本预处理进度")
        
        # 应用预处理
        df['segmented'] = df[text_columns[0]].astype(str).progress_apply(self.tokenize_text)
        
        # 将分词结果转换为字符串（用空格分隔）
        df['processed_text'] = df['segmented'].apply(lambda x: ' '.join(x))
        
        # 保存处理后的数据
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n预处理完成！结果已保存至: {output_file}")
        
        # 显示样本结果
        print("\n样本处理结果:")
        for i in range(min(3, len(df))):
            original_text = df[text_columns[0]].iloc[i]
            if len(original_text) > 80:
                original_text = original_text[:80] + "..."
            print(f"\n原始文本 ({i+1}): {original_text}")
            print(f"分词结果: {' '.join(df['segmented'].iloc[i][:10])}...")
        
        return df

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 显示环境信息
    print(f"Python版本: {sys.version}")
    print(f"Pandas版本: {pd.__version__}")
    
    # 输入文件路径
    input_file = 'sina_news.csv'  # 替换为你的数据集文件路径
    
    # 停用词表路径（如果没有，可以设为None）
    stopwords_path = 'hit_stopwords.txt' if os.path.exists('hit_stopwords.txt') else None
    
    # 初始化预处理器
    preprocessor = SinaDataPreprocessor(stopwords_path)
    
    # 执行预处理并保存结果
    try:
        preprocessor.preprocess_data(input_file, output_file='data.csv')
        print("处理成功完成！")
    except Exception as e:
        print(f"处理失败: {str(e)}")
        print("建议操作:")
        print("1. 检查文件路径是否正确")
        print("2. 尝试手动打开文件确认格式")
        print("3. 将文件转换为UTF-8编码")
        print("4. 如果文件很大，尝试使用较小的样本")