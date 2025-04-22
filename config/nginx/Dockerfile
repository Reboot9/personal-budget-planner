FROM nginx:1.23.1-alpine

RUN rm /etc/nginx/conf.d/default.conf

# Copy  Nginx configuration file
COPY config/nginx/nginx.conf /etc/nginx/conf.d

CMD ["nginx", "-g", "daemon off;"]