import argparse
import io
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import pandas as pd
from .pipeline import ROOT, analyze, demo, load, COLUMNS

class Handler(BaseHTTPRequestHandler):
    def send(self,body,status=200,kind='application/json; charset=utf-8'):
        self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path=='/api/analysis': return self.send(json.dumps(self.server.analysis,ensure_ascii=False).encode())
        if self.path.startswith('/api/export/'):
            name=self.path.rsplit('/',1)[-1].removesuffix('.csv')
            if name not in COLUMNS: return self.send(b'Not found',404)
            data=self.server.analysis
            body=pd.DataFrame(data['nodes'] if name=='nodes_roles' else data[name],columns=COLUMNS[name]).to_csv(index=False).encode('utf-8-sig')
            return self.send(body,kind='text/csv; charset=utf-8')
        path=(ROOT/'web'/('index.html' if self.path=='/' else self.path.lstrip('/').split('?')[0])).resolve()
        if not path.is_relative_to(ROOT/'web') or not path.is_file(): return self.send(b'Not found',404)
        return self.send(path.read_bytes(),kind=mimetypes.guess_type(path)[0] or 'application/octet-stream')
    def do_POST(self):
        if self.path not in ['/api/upload','/api/demo']: return self.send(b'Not found',404)
        try:
            if self.path=='/api/demo': data=analyze(*demo(),synthetic=True)
            else:
                size=int(self.headers.get('Content-Length',0))
                if size>64*1024*1024: raise ValueError('Максимальный размер загрузки — 64 МБ')
                import base64
                payload=json.loads(self.rfile.read(size))
                frames=[pd.read_parquet(io.BytesIO(base64.b64decode(payload[k],validate=True))) for k in ['nodes','edges','transactions']]
                data=analyze(*frames)
            self.server.analysis=data
            self.send(json.dumps(data,ensure_ascii=False).encode())
        except Exception as e: self.send(json.dumps({'error':str(e)},ensure_ascii=False).encode(),400)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--port',type=int,default=8000); p.add_argument('--data-dir'); args=p.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    server.analysis=analyze(*(load(args.data_dir) if args.data_dir else demo()),synthetic=not bool(args.data_dir))
    print(f'Граф денег: http://127.0.0.1:{args.port}',flush=True); server.serve_forever()

if __name__=='__main__': main()
