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