# Build Stage
FROM node:20-alpine as build

WORKDIR /app

# Copy package management files
COPY frontend/package*.json ./

# Install dependencies (using npm ci for reliable builds if package-lock exists)
RUN npm install

# Copy source code
COPY frontend/ ./

# Build the Vite application for production
RUN npm run build

# Production Serving Stage
FROM nginx:alpine

# Copy built assets to Nginx html directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose Nginx default port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
