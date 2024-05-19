# Demo Heading

> Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

## Sub Heading 1

Lorem ipsum dolor sit amet consectetur adipisicing elit[^1]. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

### SubSub Heading 1.1

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium[^2],$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?
$$  
\begin{equation}
\lim_{x \to \infty} x^2_{22} - \int_{1}^{5}x\mathrm{d}x + \sum_{n=1}^{20} n^{2} = \prod_{j=1}^{3} y_{j}  + \lim_{x \to -2} \frac{x-2}{x}
\end{equation}
$$

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

$$
\begin{equation}
 \int_\alpha^\beta f'(x) \, dx=f(\beta)-f(\alpha).
\end{equation}
$$

傅立叶变换公式通常用于将一个函数（通常是一个时域函数）转换成另一个函数（通常是频域函数）。这个变换在信号处理、图像处理、物理学等领域中广泛应用。

傅立叶变换的数学表达式如下：

$$
\begin{equation}
F(\omega) = \int_{-\infty}^{\infty} f(t) \cdot e^{-i \omega t} \, dt
\end{equation}
$$

其中，

- \( F(\omega) \) 表示频域中的复数函数，它是函数 \( f(t) \) 在频率 \( \omega \) 处的振幅和相位。
- \( f(t) \) 是时域中的函数，通常表示随时间变化的信号。
- \( \omega \) 是频率，它是角频率，通常以弧度/秒为单位。
- \( e \) 是自然对数的底。
- \( i \) 是虚数单位，\( i^2 = -1 \)。
- \( \int \) 表示积分符号，将 \( f(t) \cdot e^{-i \omega t} \) 对 \( t \) 从负无穷到正无穷积分。

这个公式描述了一个连续时间的傅立叶变换。如果是离散时间的傅立叶变换，那么积分符号会被求和符号替代。

### SubSub Heading 1.2

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

```python
def batch_normalize(X, gamma, beta, epsilon=1e-5):
    mean = np.mean(X, axis=0) #[b,w,h,d]->[w,h,d]
    variance = np.var(X, axis=0)
    X_normalized = (X - mean) / np.sqrt(variance + epsilon)
    out = gamma * X_normalized + beta

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var
    return out, X_normalized, mean, variance
```

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?[^3]
Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

```c++
#include <iostream>
#include <cstdio>

using namespace std;

template<T>
T add(T lhs, T rhs) {
    return lhs + rhs;
}

int main(){
    int a = 23;
    int b = 54;
    int c = add(a,b);
    std::cout<<" c is: "<<c<<std::endl;
}
```

### SubSub Heading 1.3

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

```mermaid
sequenceDiagram
participant Alice
participant Bob
Alice->>John: Hello John, how are you?
loop Healthcheck
    John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts <br/>prevail!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

```mermaid
---
title: Animal example
---
classDiagram
    note "From Duck till Zebra"
    Animal <|-- Duck
    note for Duck "can fly\ncan swim\ncan dive\ncan help in debugging"
    Animal <|-- Fish
    Animal <|-- Zebra
    Animal : +int age
    Animal : +String gender
    Animal: +isMammal()
    Animal: +mate()
    class Duck{
        +String beakColor
        +swim()
        +quack()
    }
    class Fish{
        -int sizeInFeet
        -canEat()
    }
    class Zebra{
        +bool is_wild
        +run()
    }

```

## Sub Heading 2

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

### SubSub Heading 2.1

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

![screen](./assets/screen.png)

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium, atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

![adobe_media](./assets/adobe_media.gif)

### SubSub Heading 2.2

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

!!! error hello this is error
    Note that Tailwind’s border reset is not applied to file input buttons. This means that to add a border to a file input button, you need to explicitly set the border-style using a class like file:border-solid alongside any border-width utility:
    $ A_i=B_i+\sum_i^jD_{ij}$
    Note that Tailwind’s border reset is not applied to file input buttons.

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

!!! note hello this is note
    Note that Tailwind’s border reset is not applied to file input buttons. This means that to add a border to a file input button, you need to explicitly set the border-style using a class like file:border-solid alongside any border-width utility:
    $ A_i=B_i+\sum_i^jD_{ij}$
    Note that Tailwind’s border reset is not applied to file input buttons.

!!! warning hello this is warning
    Note that Tailwind’s border reset is not applied to file input buttons. This means that to add a border to a file input button, you need to explicitly set the border-style using a class like file:border-solid alongside any border-width utility:
    $ A_i=B_i+\sum_i^jD_{ij}$
    Note that Tailwind’s border reset is not applied to file input buttons.
    ```python
      import pandas as pd
      def function(a,b):
        return a+b
    ```
!!! info hello this is info
    Note that Tailwind’s border reset is not applied to file input buttons. This means that to add a border to a file input button, you need to explicitly set the border-style using a class like file:border-solid alongside any border-width utility:
    $ A_i=B_i+\sum_i^jD_{ij}$
    Note that Tailwind’s border reset is not applied to file input buttons.

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

!!! tip hello this is tip
    Note that Tailwind’s border reset is not applied to file input buttons. This means that to add a border to a file input button, you need to explicitly set the border-style using a class like file:border-solid alongside any border-width utility:
    $ A_i=B_i+\sum_i^jD_{ij}$
    Note that Tailwind’s border reset is not applied to file input buttons.
    ```python
      import pandas as pd
      def function(a,b):
        return a+b
    ```

### SubSub Heading 2.3

Lorem ipsum dolor sit amet consectetur adipisicing elit. Eligendi ipsam, pariatur illum odit beatae numquam ab fuga voluptas sequi maxime praesentium,$A=\sum_{i=0}^{i<100}B_i + \int_{i=0}^{i=100}\frac{f(x)}{g(x)}dx$ atque amet doloribus nostrum, eveniet aliquam perferendis perspiciatis repellat?

| col1 | col2 | col3 | col4 |
|-|-|-|-|
|row1|row1|row1|row1|
|row1|row1|row1|row1|
|row1|row1|row1|row1|
|row1|row1|row1|row1|

[^1]: Note that Tailwind’s border reset is not applie Note that Tailwind’s border reset is not applie Note that Tailwind’s border reset is not applie
[^2]: Note that Tailwind’s border reset is not applie
[^3]: Note that Tailwind’s border reset is not applie Note that Tailwind’s border reset is not applie
