FROM node:18 AS builder

WORKDIR /application
COPY ui/frontend .
RUN yarn install

# https://stackoverflow.com/a/69746937/905845
ENV NODE_OPTIONS --openssl-legacy-provider
ENV PPL_BASE_PATH /ppl
RUN npx nuxt build
RUN npx nuxt export


FROM nginx:1.21

ADD ui/prod-env/frontend-and-fanout/nginx.conf /etc/nginx/nginx.conf
COPY --from=builder /application/dist /application/ppl
