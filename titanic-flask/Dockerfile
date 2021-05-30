FROM continuumio/miniconda3

WORKDIR /app

COPY environment.yml .
RUN conda env create --file environment.yml

# Make RUN commands use the new environment:
# solution from https://pythonspeed.com/articles/activate-conda-dockerfile/
SHELL ["conda", "run", "-n", "titanic-flask", "/bin/bash", "-c"]

COPY . .

EXPOSE 5000
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "titanic-flask", "python", "app.py"]