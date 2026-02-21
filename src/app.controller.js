import dotenv from 'dotenv'
import path from "path";
import cors from 'cors';
import { pipeline, env } from '@huggingface/transformers';

import { InferenceClient } from '@huggingface/inference';



const envPath = path.resolve("src","config",".env")

dotenv.config({ path: envPath })

const hf = new InferenceClient(process.env.HF_API_TOKEN);



const port = process.env.PORT || 5000;

const bootstrap  = async (app,express)=>{


app.use(cors());
app.use(express.json());

app.get('/', (req, res) => {
  res.send('Hello World')
})

/////////////////////////////////////TEST//////////////////////////////////////
env.allowLocalModels = false; // Forces it to use the internet, not your server's disk
env.useBrowserCache = false;  // Prevents permission issues in the .cache folder
env.allowRemoteModels = true; 

let detector;

const loadModel = async () => {
  try {
    console.log("Testing HF connection...")

    // simple test call
    const result = await hf.textClassification({
      model: "openai-community/roberta-base-openai-detector",
      inputs: "Hello world"
    })

    console.log("✅ HF API working:", result)
  } catch (err) {
    console.error("❌ HF API error:", err)
  }
}

loadModel()
/////////////////////////////////////////////////////////////////////////////////////////////////////

app.listen(port, () => {
  console.log(`Server is running on ${port}`)
})

}

export default bootstrap