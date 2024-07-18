import os
import argparse
import sys
sys.path.append('/mnt/yizhou/Shenzhen_GLM_Project/Core/')


from Bio import SeqIO
import random

from Utils.utils import get_data_logger


logger = get_data_logger('fasta_log')


def extract_random_fasta(input_file, output_file, percentage):
    # 读取输入FASTA文件中的所有序列
    sequences = list(SeqIO.parse(input_file, "fasta"))
    logger.info(f'The fasta file has {len(sequences)} sequences')
    
    if not sequences:
        logger.error(f"No sequences found in the input file {input_file}")
        return
    
    # 计算要提取的序列数
    num_sequences_to_extract = int(len(sequences) * percentage / 100)
    logger.info(f'Extracting {num_sequences_to_extract} sequences')
    
    # 随机选择序列
    random_sequences = random.sample(sequences, num_sequences_to_extract)
    
    if not random_sequences:
        logger.error("No sequences were selected. Check the percentage value.")
        return
    
    # 将随机选择的序列写入输出FASTA文件
    if output_file:
        SeqIO.write(random_sequences, output_file, "fasta")
        logger.info(f'Sequences written to {output_file}')
    else:
        # 如果没有指定输出文件路径，则输出到标准输出
        SeqIO.write(random_sequences, sys.stdout, "fasta")



def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description="Data processing script to extract random sequences from a FASTA file.")
    parser.add_argument("-i", "--input_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("-p", "--percentage", type=float, required=True, help="Percentage of sequences to extract.")
    parser.add_argument("-o", "--output_file", type=str, default=None,required=False, help="Path to the output FASTA file.")
    # 解析参数
    args = parser.parse_args()

    # 调用函数处理 FASTA 文件
    extract_random_fasta(args.input_file, args.output_file, args.percentage)

if __name__ == "__main__":
    main()
