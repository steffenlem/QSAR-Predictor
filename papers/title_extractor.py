#!/usr/bin/env python3

import sys
import shutil

from PyPDF2 import PdfFileReader

def main():
  path = sys.argv[1]
  title = get_title(path)
  if title.strip():
    shutil.move(path, f"{title}.pdf")

def get_title(path):
  with open(path, 'rb') as file:
    reader = PdfFileReader(file)
    return reader.getDocumentInfo().title

if __name__ == "__main__":
  main()
