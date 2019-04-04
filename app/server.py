from starlette.applications import Starlette
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
import uvicorn, aiohttp, asyncio
from io import BytesIO

from fastai import *
from fastai.vision import *

# export_file_url = 'https://www.dropbox.com/s/v6cuuvddq73d1e0/export.pkl?raw=1'
export_file_url = 'https://drive.google.com/uc?export=download&id=1B4vDCZ6Pz8GkCzNT0ueYfmZZEVYWB3dr'
export_file_name = 'senadores.pkl'

classes = ['Acir_Gurgacz', 'Aecio_Neves', 'Aloysio_Nunes_Ferreira', 'Alvaro_Dias', 'Ana_Amelia', 'Angela_Portela', 'Antonio_Anastasia', 
           'Antonio_Carlos_Valadares', 'Armando_Monteiro', 'Ataides_Oliveira', 'Benedito_de_Lira', 'Blairo_Maggi', 'Cassio_Cunha_Lima', 
           'Cidinho_Santos', 'Ciro_Nogueira', 'Cristovam_Buarque', 'Dalirio_Beber', 'Dario_Berger', 'Davi_Alcolumbre', 'Edison_Lobao', 
           'Eduardo_Amorim', 'Eduardo_Braga', 'Elmano_Ferrer', 'Eunicio_Oliveira', 'Fatima_Bezerra', 'Fernando_Bezerra_Coelho', 
           'Fernando_Collor', 'Flexa_Ribeiro', 'Garibaldi_Alves_Filho', 'Gladson_Cameli', 'Gleisi_Hoffmann', 'Helio_Jose', 'Humberto_Costa', 
           'Ivo_Cassol', 'Jader_Barbalho', 'Joao_Alberto_Souza', 'Joao_Capiberibe', 'Jorge_Viana', 'Jose_Agripino', 'Jose_Anibal', 
           'Jose_Maranhao', 'Jose_Medeiros', 'Jose_Pimentel', 'Katia_Abreu', 'Lasier_Martins', 'Lidice_da_Mata', 'Lindbergh_Farias', 
           'Lucia_Vania', 'Magno_Malta', 'Maria_do_Carmo_Alves', 'Marta_Suplicy', 'Omar_Aziz', 'Otto_Alencar', 'Paulo_Bauer', 'Paulo_Paim', 
           'Paulo_Rocha', 'Pedro_Chaves', 'Raimundo_Lira', 'Randolfe_Rodrigues', 'Regina_Sousa', 'Reguffe', 'Renan_Calheiros', 
           'Ricardo_Ferraco', 'Roberto_Muniz', 'Roberto_Requiao', 'Romario', 'Romero_Juca', 'Ronaldo_Caiado', 'Rose_de_Freitas', 
           'Sergio_Petecao', 'Simone_Tebet', 'Tasso_Jereissati', 'Telmario_Mota', 'Valdir_Raupp', 'Vanessa_Grazziotin', 
           'Vicentinho_Alves', 'Waldemir_Moka', 'Wellington_Fagundes', 'Wilder_Morais', 'Zeze_Perrella']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f: f.write(data)

async def setup_learner():
    await download_file(export_file_url, path/export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()

@app.route('/')
def index(request):
    html = path/'view'/'index.html'
    return HTMLResponse(html.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    data = await request.form()
    img_bytes = await (data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction,act,prob = learn.predict(img)[0]
    return JSONResponse({'result': str(prob)})

if __name__ == '__main__':
    if 'serve' in sys.argv: uvicorn.run(app=app, host='0.0.0.0', port=5042)
