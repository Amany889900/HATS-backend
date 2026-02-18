import dotenv from 'dotenv'
import path from "path";


const envPath = path.resolve("src","config",".env")

dotenv.config({ path: envPath })


const port = process.env.PORT || 5000;

const bootstrap  = async (app,express)=>{



app.use(express.json());

app.get('/', (req, res) => {
  res.send('Hello World')
})

app.listen(port, () => {
  console.log(`Server is running on ${port}`)
})

}

export default bootstrap