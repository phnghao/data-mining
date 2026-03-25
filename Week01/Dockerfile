FROM python:3.10-slim
WORKDIR /week01
COPY ./requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --default-timeout=1500\
 -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
CMD ["python", "week01.py"]