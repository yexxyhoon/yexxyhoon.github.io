# 문자열 formatting


```python
a = 100
```

* 문자열 포멧코드
    + %s: 문자열   
    + %d: 정수  
    + %f: 부동소수점(%0.2f: 소수점 2자리까지 표시)  
    + %%: "%"라는 문자  


```python
print("a =  %f kg" %a)
```

    a =  100.000000 kg
    


```python
print("a = %0.2f kg" %a)
```

    a = 100.00 kg
    


```python
print("a = %d kg" %a)
```

    a = 100 kg
    

문자열 formatting의 핵심은 "줄맞춤"이다.  
오른쪽 줄맞춤이 기본이다.


```python
print("a = %20f kg" %a)
```

    a =           100.000000 kg
    

실제 자릿수보다 짧은 것이 들어오면 무시한다.


```python
print("a = %4f kg" %a)
```

    a = 100.000000 kg
    


```python
print("a = %4.2f kg" %a)
```

    a = 100.00 kg
    


```python
print("a = %s & b = %d"%("xyz",100))
```

    a = xyz & b = 100
    


```python
print("a =  %f %%" %10)
```

    a =  10.000000 %
    

# 고급 formatting
* format 함수  
    x="{0}다리는 {1}개"  
    x.format("사람",2)


```python
print("{} and {}".format(10,20))
```

    10 and 20
    


```python
print("{1} and {0}".format(10,20))
```

    20 and 10
    


```python
print("{1:d} and {0:0.2f}".format(10,20))
```

    20 and 10.00
    


```python
print("{1:6d} and {0:10.2f}".format(10,20))
```

        20 and      10.00
    

고급 formatting의 핵심은 "순서바꿈"과 "이름지정"이다.  
왼쪽 줄맞춤이 기본이다.


```python
print("{0:10s}".format("ABC"))
```

    ABC       
    


```python
print("{0:>10s}".format("ABC"))
```

           ABC
    


```python
print("{0:^10s}".format("ABC"))
```

       ABC    
    


```python
print("{0:*<10s}".format("ABC"))
```

    ABC*******
    


```python
print("{0:0^10s}".format("ABC"))
```

    000ABC0000
    


```python
print("{0:+0.2f}".format(1.23456789))
```

    +1.23
    


```python
print("{0:+10.2f}".format(1.23456789))
```

         +1.23
    


```python
print("{0:+10.2%}".format(1.23456789))
```

      +123.46%
    


```python

```
