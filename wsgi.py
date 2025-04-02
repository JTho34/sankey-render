from app import server as application

# Simple fichier wsgi pour Gunicorn
# La variable 'application' est ce que Gunicorn va chercher

if __name__ == "__main__":
    application.run()