   clear all
   close all
   clc
   Hbo = 2 * tf([-2 1],[0.4 2.2 1 0]);
   figure(2); clf; nyquist(Hbo); grid; title('Diagramme de Nyquist');