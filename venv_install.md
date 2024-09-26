# Passo a passo para instalar o ambiente virtual
Instale a biblioteca venv para criar o ambiente virtual utilizando a maneira de sua preferências, no caso de Arch Linux `sudo pacman -S python-virtualenv`.

No diretório Root do repositório e com o ambiente instalado, no no terminal digite `python -m venv venv` para criar as pastas de venv e .venv.

Ainda no root insira `source venv/bin/activate`, o que irá ativar o ambiente virtual e rode `pip install -r requirements.txt` para instalar as dependências.

Para sair do ambiente virtual basta digitar deactivate no terminal.