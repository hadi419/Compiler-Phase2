#include <stdio.h>

int fib(int n);

int internal_fib(int n)
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


int main()
{
  for (int i = 0; i <= 10; i++)
  {
    int f1 = fib(i);
    int f2 = internal_fib(i);

    if (f1 != f2)
      return 1;

    printf("fib(%d) -> %d\n", i, f1);
  }

  return 0;
}