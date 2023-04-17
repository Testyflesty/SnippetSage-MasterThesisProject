FROM node:14 AS frontend

WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .

RUN npm run build

FROM python:3.7 AS backend

WORKDIR /app/backend

COPY backend/requirements.txt ./
RUN pip install -r requirements.txt
COPY backend/ .

WORKDIR /app/baseline

COPY baseline/ .

WORKDIR /app

COPY --from=frontend /app/frontend/dist /app/backend/static
COPY --from=frontend /app/frontend/dist /app/baseline/static

EXPOSE 5005 5173 5055 8080

CMD ["bash", "-c",  "cd /app/backend & rasa run -m models --enable-api --cors '*' --port 5005 & rasa run actions --cors '*' & python baseline/app.py & cd /app/frontend && npm run dev"]