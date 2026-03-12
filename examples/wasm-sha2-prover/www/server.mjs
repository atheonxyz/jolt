// Simple HTTP server with COOP/COEP headers and directory index.js fallback
import { createServer } from 'http';
import { readFile, stat } from 'fs/promises';
import { join, extname } from 'path';
import { fileURLToPath } from 'url';

const __dirname = fileURLToPath(new URL('.', import.meta.url));
const PORT = parseInt(process.argv[2] || process.env.PORT || '8080', 10);

const MIME = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.wasm': 'application/wasm',
    '.json': 'application/json',
    '.css': 'text/css',
    '.bin': 'application/octet-stream',
    '.elf': 'application/octet-stream',
    '.wgsl': 'text/plain',
};

const server = createServer(async (req, res) => {
    let url = new URL(req.url, `http://localhost:${PORT}`).pathname;

    let filePath = join(__dirname, url);
    const headers = {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
    };

    try {
        let s = await stat(filePath);
        // If it's a directory, try serving the main JS module (for wasm-bindgen-rayon)
        if (s.isDirectory()) {
            // Try package.json → main field, or fallback to index.html/index.js
            try {
                const pkg = JSON.parse(await readFile(join(filePath, 'package.json'), 'utf-8'));
                if (pkg.module) filePath = join(filePath, pkg.module);
                else if (pkg.main) filePath = join(filePath, pkg.main);
                else filePath = join(filePath, 'index.html');
            } catch {
                // No package.json — try index.html
                filePath = join(filePath, 'index.html');
            }
            s = await stat(filePath);
        }

        const ext = extname(filePath);
        const mime = MIME[ext] || 'application/octet-stream';
        const data = await readFile(filePath);
        res.writeHead(200, { ...headers, 'Content-Type': mime, 'Content-Length': data.length });
        res.end(data);
    } catch {
        res.writeHead(404, headers);
        res.end('Not found');
    }
});

server.listen(PORT, () => {
    console.log(`Server running at http://localhost:${PORT}/`);
});
