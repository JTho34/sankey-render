from app import server

# C'est ce fichier que gunicorn va chercher
# Nous exposons simplement la variable server de app.py

if __name__ == "__main__":
    server.run() 