# build stage
FROM oven/bun:1.1.13 AS builder

WORKDIR /app

# copy only package files first for better caching
COPY package.json bun.lock ./
RUN bun install

# copy the rest of the frontend source
COPY . .

# build the frontend
RUN bun run build

# production stage
FROM oven/bun:1.1.13 AS runner

WORKDIR /app

# copy server code and built frontend from builder
COPY --from=builder /app/server ./server
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./
COPY --from=builder /app/bun.lock ./

# expose the port the server runs on
EXPOSE 3000

# start the Bun server
CMD ["bun", "server/index.ts"]
