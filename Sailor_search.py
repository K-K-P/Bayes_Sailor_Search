"""SAVING SHIPWRECKED SAILORS WITH BAYES’ RULE
Using OpenCV library we will create a software for the searching procedure.
On map, we place 3 squares - in one of them, there's a missing sailor - user needs to find a missing person, using
Bayes' theorem or his own 'guts'"""

import sys
import random
import itertools
import numpy as np
import cv2 as cv

MAP_FILE: str = 'cape_python.png'  # background image, here in the same directory
# Define 3 squares - for each of them two corners are enough - (UpperLeft-X, UpperLeft-Y, LowerRight-X, LowerRight-Y)
SA1_CORNERS: tuple = (130, 265, 180, 315)
SA2_CORNERS: tuple = (80, 255, 130, 305)
SA3_CORNERS: tuple = (105, 205, 155, 255)


class Search:
    """Bayesian Search & Rescue game"""
    def __init__(self, name: str):
        self.name: str = name
        self.img: np.ndarray = cv.imread(MAP_FILE, cv.IMREAD_COLOR)  # IMREAD_COLOR to load image in color mode / gray by default
        if self.img is None:
            print(f'Could not load a {MAP_FILE}', file=sys.stderr)  # If image not loaded, print message to std...
            sys.exit(1)  # ...and exit
        self.cur_area: int = 0  # number of currently searched area
        self.sailor_coor: list = [0, 0]  # sailor's position coordinates
        # Create three search areas, given the width and and height of the square
        self.sa1: np.ndarray = self.img[SA1_CORNERS[1]:SA1_CORNERS[3], SA1_CORNERS[0]:SA1_CORNERS[2]]  # First y then x coordinates!
        self.sa2: np.ndarray = self.img[SA2_CORNERS[1]:SA2_CORNERS[3], SA2_CORNERS[0]:SA2_CORNERS[2]]
        self.sa3: np.ndarray = self.img[SA3_CORNERS[1]:SA3_CORNERS[3], SA3_CORNERS[0]:SA3_CORNERS[2]]
        # Setting probabilities for all search areas - this and sep could provided by a specialized tool
        self.p1: float = 0.2
        self.p2: float = 0.5
        self.p3: float = 0.3
        # Setting sep - search effectiveness probability
        self.sep1: float = 0
        self.sep2: float = 0
        self.sep3: float = 0

    def draw_map(self, last_known: tuple):
        """Draw and display map with a legend and a scale"""
        cv.line(self.img, (20, 370), (70, 370), (0, 0, 0), 2)  # img, starting coord, end coord, color RGB, thck in px
        cv.putText(self.img, '0', (8, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))  # placing the legend, left to line
        cv.putText(self.img, '50 Nautical miles', (71, 370), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))  # as above, but right
        cv.rectangle(self.img, (SA1_CORNERS[0], SA1_CORNERS[1]), (SA1_CORNERS[2], SA1_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '1', (SA1_CORNERS[0] + 3, SA1_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.rectangle(self.img, (SA2_CORNERS[0], SA2_CORNERS[1]), (SA2_CORNERS[2], SA2_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '2', (SA2_CORNERS[0] + 3, SA2_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))
        cv.rectangle(self.img, (SA3_CORNERS[0], SA3_CORNERS[1]), (SA3_CORNERS[2], SA3_CORNERS[3]), (0, 0, 0), 1)
        cv.putText(self.img, '3', (SA3_CORNERS[0] + 3, SA3_CORNERS[1] + 15), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 0))

        cv.putText(self.img, '+', last_known, cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
        cv.putText(self.img, '+ = LAST KNOWN POSITION', (274, 355), cv.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))  # BGR color
        cv.putText(self.img, '* = ACTUAL POSITION', (275, 370), cv.FONT_HERSHEY_PLAIN, 1, (255, 0, 0))

        cv.imshow('Search Area', self.img)
        cv.moveWindow('Search Area', 750, 50)  # move window to the position on the screen when invoked width x height
        cv.waitKey(500)  # wait time (in ms) after any key is pressed

    def sailor_location(self, num_search_areas: int) -> tuple:
        """Return the position of the missing sailor"""
        self.sailor_coor[0] = np.random.choice(self.sa1.shape[1], 1)[0]  # since all search area are 50 x 50 px
        self.sailor_coor[1] = np.random.choice(self.sa1.shape[0], 1)[0]  # we can the initial coor from any of 3 areas
        # index after random choice because it returns array instead of a number
        area: int = int(random.triangular(1, num_search_areas + 1))  # randomly pick searched area (here 1 to 3 (4 excluded))
        #  Place sailor in one of the areas:
        if area == 1:
            x: int = self.sailor_coor[0] + SA1_CORNERS[0]
            y: int = self.sailor_coor[1] + SA1_CORNERS[1]
            self.cur_area = 1
        if area == 2:
            x: int = self.sailor_coor[0] + SA2_CORNERS[0]
            y: int = self.sailor_coor[1] + SA2_CORNERS[1]
            self.cur_area = 2
        if area == 3:
            x: int = self.sailor_coor[0] + SA3_CORNERS[0]
            y: int = self.sailor_coor[1] + SA3_CORNERS[1]
            self.cur_area = 3

        if x and y:
            return x, y

    def calc_search_effectiveness(self) -> None:
        """Set search effectiveness for every search area"""
        self.sep1 = random.uniform(0.2, 0.9)
        self.sep2 = random.uniform(0.2, 0.9)
        self.sep3 = random.uniform(0.2, 0.9)

    def search(self, area_num: int, area_array: np.ndarray, effectiveness_prob: float) -> [str, list]:
        """Return search results and coordinates"""
        y_range = range(area_array.shape[0])  # get the x, y ranges from sa1 / sa2 / sa3 areas
        x_range = range(area_array.shape[1])
        coords = list(itertools.product(x_range, y_range))  # get all possible permutations of coordinates
        random.shuffle(coords)
        coords = coords[:int(len(coords) * effectiveness_prob)]  # effectiveness affects the number of coords to be searched
        loc_actual = (self.sailor_coor[0], self.sailor_coor[1])  # get the actual sailor's position
        if area_num == self.cur_area and loc_actual in coords:
            return f'Found in area {area_num}', coords
        else:
            return 'Nothing here', coords

    def revise_target_probabilities(self) -> None:
        """Update area probabilities basing on search efectiveness"""
        denominator: float = self.p1 * (1 - self.sep1) + self.p2 * (1 - self.sep2) + self.p3 * (1 - self.sep3)  # split it
        self.p1 = self.p1 * (1 - self.sep1) / denominator
        self.p2 = self.p2 * (1 - self.sep2) / denominator
        self.p3 = self.p3 * (1 - self.sep3) / denominator


def create_menu(search_num: int = 1) -> None:
    """Print out the menu of choices"""
    print(f'Search {search_num}')
    print("""
    Choose the area to be searched:
    
    0 - Quit
    1 - Search Area 1 twice
    2 - Search Area 2 twice
    3 - Search Area 3 twice
    4 - Search Areas 1 and 2
    5 - Search Areas 1 and 3
    6 - Search Areas 2 and 3
    7 - Restart
    """)


def main():
    app = Search('Python Search Game')
    app.draw_map(last_known=(160, 290))
    sailor_x, sailor_y = app.sailor_location(num_search_areas=3)
    print('~' * 60)
    print('Initial probabilities are:')
    print(f'P1: {app.p1}, P2: {app.p2}, P3: {app.p3}')
    search_num: int = 1
    while True:
        app.calc_search_effectiveness()
        create_menu(search_num=search_num)
        choice: str = input('What do you choose to do?\n')
        print(choice)
        # Menu logic goes here:
        if choice == '0':
            sys.exit()
        elif choice == '1':
            result1, coords1 = app.search(1, app.sa1, app.sep1)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(1, app.sa1, app.sep1)  # the same for 2nd search
            app.sep1: int = (int(len(set(coords1 + coords2)))) / len(app.sa1)**2  # divide len o searched points list by
            # overal number of points to be searched (here 50 x 50, hence 50^2)
            app.sep2: int = 0  # other two weren't looked through, so set sep to 0
            app.sep3: int = 0
        elif choice == '2':
            result1, coords1 = app.search(2, app.sa2, app.sep2)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(2, app.sa2, app.sep2)  # the same for 2nd search
            app.sep2: int = (int(len(set(coords1 + coords2)))) / len(app.sa2)**2  # divide len o searched points list by
            # overal number of points to be searched (here 50 x 50, hence 50^2)
            app.sep1: int = 0  # other two weren't looked through, so set sep to 0
            app.sep3: int = 0
        elif choice == '3':
            result1, coords1 = app.search(3, app.sa3, app.sep3)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(3, app.sa3, app.sep3)  # the same for 2nd search
            app.sep3: int = (int(len(set(coords1 + coords2)))) / len(app.sa3)**2  # divide len o searched points list by
            # overal number of points to be searched (here 50 x 50, hence 50^2)
            app.sep2: int = 0  # other two weren't looked through, so set sep to 0
            app.sep1: int = 0
        elif choice == '4':
            result1, coords1 = app.search(1, app.sa1, app.sep1)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(2, app.sa2, app.sep2)  # the same for 2nd search
            app.sep3: int = 0
        elif choice == '5':
            result1, coords1 = app.search(1, app.sa1, app.sep1)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(3, app.sa3, app.sep3)  # the same for 2nd search
            app.sep2: int = 0
        elif choice == '6':
            result1, coords1 = app.search(2, app.sa2, app.sep2)  # get the coordinates looked through in 1st search
            result2, coords2 = app.search(3, app.sa3, app.sep3)  # the same for 2nd search
            app.sep1: int = 0
        elif choice == '7':
            main()
        else:
            print('Please write the correct number to proceed')
            continue
        app.revise_target_probabilities()  # recalculate the probabilities after the search was done
        print(f'Search {search_num} Result 1: {result1}')
        print(f'Search {search_num} Result 2: {result2}')
        print(f'E1: {app.sep1:.2f}, E2: {app.sep2:.2f}, E3: {app.sep3:.2f}')
        if result1 == 'Nothing here' and result2 == 'Nothing here':
            print(f'Nothing found. New probabilities for search {search_num + 1} are:')
            print(f'P1: {app.p1}, P2: {app.p2}, P3: {app.p3}')
        else:
            cv.destroyAllWindows()  # Close the search window
            cv.circle(app.img, (sailor_x, sailor_y), 3, (255, 0, 0), -1)
            cv.imshow('Python Search Game', app.img)
            cv.moveWindow('Python Search Game', 50, 600)
            cv.waitKey(1500)
            while True:
                play_again: str = input('Do you wish to play again? Y/N?\n')
                if play_again.lower().strip() == 'y':
                    cv.destroyAllWindows()  # close the window with sailor's marked position
                    main()
                elif play_again.lower().strip() == 'n':
                    sys.exit()
                else:
                    print('Wrong input')

        search_num += 1


if __name__ == '__main__':
    main()

