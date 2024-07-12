int add(int a, int b);
int addOne();

int add(int a, int b)
{
    int local1 = 10;
    int local2;

    return a + b;
}

int caller()
{
    return add(1, 2) + 3;
}

int caller2()
{
    return add(1, 2) + add(3, 4);
}

int infiniteLoop()
{
    return infiniteLoop();
}

void doSomething()
{

}

void inifiniteLoop2()
{
    for (;;)
        doSomething();
}

void forLoop()
{
    for (int i = 0; i < 10; i++)
        fib(addOne(i));
}

int addOne(int a)
{
    int expr1 = -20 * !1 * ~0;
    int expr2 = 10 << 20 >> 30;
    int expr3 = 0 || 1 && 1;
    int expr4 = 2^10 & 2 | 3;

    return a + 1;
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

void interpret(int opcode)
{
    switch (opcode)
    {
        case 0b00:
            doSomething();
            break;

        case 0b01:
            doSomething();
        case 0b10:
            inifiniteLoop();
            break;

        default:
            interpret(opcode + 1);
            break;
    }
}

enum OsType;
enum Animal;

int a = 10 + (2 + 3) * 4;
int b;

void setter()
{
    b = a;
}

enum OsType
{
    Linux,
    Mac = 4,
    Windows = 6,
    Windows = 7
} myOs;

enum OsType yourOs = Linux;

enum Animal
{
    Dog,
    Cat
};

enum Animal animal = Dog;

int globalArray[2][3] = { { 1, 2, 3 }, { 4, 5, 6 } };

int valueFromArray(int index)
{
    int array[3] = {
        1, 2, 3
    };

    return array[index];
}
