FROM node:latest

WORKDIR /visualizer

COPY package.json .

RUN npm install

EXPOSE 3000

ENTRYPOINT ["npm", "start"]
