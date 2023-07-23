import cv2


class Box:
    def __init__(self, tl, br, color):
        self.tl = tl.copy()
        self.br = br.copy()
        self.w = self.br[0] - self.tl[0]
        self.h = self.br[1] - self.tl[1]
        self.center = self.calc_center()
        self.color = color

    def draw(self, frame):
        cv2.rectangle(frame, 
                      list(map(int, self.tl)), 
                      list(map(int, self.br)), 
                      color=self.color, thickness=3)

    def calc_center(self):
        x = self.tl[0] + self.w / 2
        y = self.tl[1] + self.h / 2
        return [x,y]

    def set_center(self, center):
        dx = center[0] - self.center[0]
        dy = center[1] - self.center[1]

        self.tl[0] += dx
        self.tl[1] += dy
        self.br[0] += dx
        self.br[1] += dy

        self.center = center

    def reset_box(self, x, y, w, h):
        self.w = w
        self.h = h

        self.tl[0] = x
        self.tl[1] = y
        self.br[0] = x + w
        self.br[1] = y + h

        self.center = self.calc_center()

    def move_up(self, pixels):
        self.tl[1] -= pixels
        self.br[1] -= pixels
        self.center[1] -= pixels
