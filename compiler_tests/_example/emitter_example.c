int add(int a, int b);

int add(int a, int b)
{
    int c;
    c = a + b;

    return c;
}

int caller()
{
    return add(1, 2) + 3;
}

int caller2()
{
    return add(1, 2) + add(3, 4);
}

int recursiveFib(int n)
{
    if (n < 2)
        return n;

    return recursiveFib(n - 1) + recursiveFib(n - 2);
}

void doSomething()
{

}

void doSomethingWith(int n)
{

}

int fib(int n);

void forLoop(int n)
{
    for (int i = 0; i < n; i++)
        if (i % 2 == 0)
            doSomethingWith(fib(i));
}

int forLoop2(int n)
{
    int i;
    for (i = 0; i < n; i++)
        doSomething();

    return i;
}

void forLoop3()
{
    for (;;)
        doSomething();
}

int subScope()
{
    int x = 0;
    {
        int x = 1;
    }

    return x;
}

int eq(int a, int b) { return a == b; }

int ne(int a, int b) { return a != b; }

int lt(int a, int b) { return a < b; }

int gt(int a, int b) { return a > b; }

int le(int a, int b) { return a <= b; }

int ge(int a, int b) { return a >= b; }

int xr(int a, int b) { return a ^ b; }

int bw(int n) { return ~n; }

int flip(int n) { return !n; }

int neg(int n) { return -n; }

int pos(int n) { return +n; }

int inc(int n) { return ++n; }

int dec(int n) { return --n; }

int inc2(int n) { return n++; }

int dec2(int n) { return n--; }

int implicitReturn()
{

}

int foo(int n)
{
    int k;

    return k = n += 2;
}

int min(int a, int b)
{
    if (a < b)
        return a;

    return b;
}

int fib(int n)
{
    int a = 0;
    int b = 1;
    int c;
    int i = 2;

    if (n == 0) return a;
    if (n == 1) return b;

    while (i++ <= n)
    {
        c = a + b;
        a = b;
        b = c;
    }

    return b;
}

int a = 10 + 2 * 3;
int b;

void setter()
{
    b = a;
}

int logicalAnd(int a, int b)
{
    return a && b;
}

int logicalOr(int a, int b)
{
    return a || b;
}

int intFormat()
{
    return 0x000023 + 0b0101 == 35 + 5;
}

int array[2][2] = { { 1, 2 }, { 3 } };

int indexArray(int n)
{
    return array[n];
}

int indexArray2(int n)
{
    int localArray[2][2] = { { 1, 2 }, { 3 } };

    return localArray[n];
}

enum Animal
{
    Dog,
    Cat,
    Drake = 5,
    Snake
};

enum Animal animal1 = Drake;
enum { X } animal2 = X;

void f(int n)
{
    switch (n)
    {
        case 0:
            return;

        case 1:
        case 2:
            doSomething();
            break;

        case 3:
            doSomethingWith(1);
        case 4:
            doSomethingWith(2);
            break;

        default:
            doSomethingWith(3);
            break;
    }
}
