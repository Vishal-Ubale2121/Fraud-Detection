from django.shortcuts import render

def home(request):
    return render(request, 'AI_Fraud_Detection/test.html')

def about(request):
    return render(request, 'AI_Fraud_Detection/about.html')

def test(request):
    return render(request, 'AI_Fraud_Detection/test.html')
