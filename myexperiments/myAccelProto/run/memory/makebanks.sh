#!/bin/bash

# 파일을 생성할 디렉토리 (필요시 수정)
output_dir="./"

# 0부터 191까지의 파일 생성
for i in $(seq 0 299); do
  touch "${output_dir}${i}"
done

