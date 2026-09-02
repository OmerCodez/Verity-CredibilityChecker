# Verity

## Run locally

1. Create an NVIDIA Build API key at https://build.nvidia.com/.
2. Copy `.env.example` to `.env` and replace the placeholder with that key.
3. Start the site:

   ```powershell
   npm start
   ```

4. Open `http://localhost:3000`.

`server.js` downloads the submitted article and asks NVIDIA's model to analyse that content. It keeps your NVIDIA API key on the server. Do not put the key in `Index.html`, because every visitor could extract it.

The default model is NVIDIA Nemotron 3.5 Lightning. Override `NVIDIA_MODEL` if you select another NVIDIA Build model.
