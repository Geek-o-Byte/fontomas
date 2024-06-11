from fastapi import FastAPI, Request, Depends, HTTPException, status, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from typing import Optional
import os
from models import User, Base, engine  # Импортируем User и Base напрямую из models
import shutil
from fontTools.ttLib import TTFont

app = FastAPI()

app.add_middleware(SessionMiddleware, secret_key="your-secret-key")

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_db():
    db = SessionLocal()  # Теперь SessionLocal доступен прямо из этого файла
    try:
        yield db
    finally:
        db.close()


def get_password_hash(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


@app.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...),
                   db: Session = Depends(get_db)):
    hashed_password = get_password_hash(password)
    db_user = User(username=username, email=email, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)


@app.post("/login")
async def login(request: Request, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Incorrect username or password")
    request.session['user_id'] = user.id
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/logout")
async def logout(request: Request):
    request.session.pop('user_id', None)
    return RedirectResponse(url="/", status_code=status.HTTP_302_FOUND)


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    fonts = os.listdir("static/fonts")
    fonts = [font for font in fonts if font.endswith(('.ttf', '.otf'))]
    return templates.TemplateResponse("index.html", {"request": request, "fonts": fonts})


def get_font_info(file_path):
    font = TTFont(file_path)
    name_records = font['name'].names
    font_info = {
        'name': '',
        'style': '',
        'weight': '',
        'creators': []
    }
    for record in name_records:
        name = record.string.decode(record.getEncoding())
        if record.nameID == 1:
            font_info['name'] = name
        elif record.nameID == 2:
            font_info['style'] = name
        elif record.nameID == 4:
            font_info['weight'] = name
        elif record.nameID == 9:
            font_info['creators'].append(name)
    return font_info


@app.get("/font/{font_name}", response_class=HTMLResponse)
async def font_page(request: Request, font_name: str):
    font_path = f"static/fonts/{font_name}"
    if not os.path.exists(font_path):
        raise HTTPException(status_code=404, detail="Font not found")

    font_info = get_font_info(font_path)
    font_info['url'] = f"/static/fonts/{font_name}"

    return templates.TemplateResponse("font_page.html", {"request": request, "font_info": font_info})


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})
