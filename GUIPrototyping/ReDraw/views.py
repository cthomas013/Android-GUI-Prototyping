from django.shortcuts import render, redirect

def index(request):
    return render(request, "index.html")

def images(request):
    return redirect("/static" + request.path)

def about(request):
    return render(request, "about.html")

def gallery(request): 
    return render(request, "gallery.html")
