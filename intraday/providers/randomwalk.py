import math
import random
from typing import NamedTuple, Union, Optional, Tuple
from numbers import Real
import arrow
import numpy as np
from datetime import datetime, date, timedelta, timezone

from intraday.provider import Provider, Trade


class RandomWalkProvider(Provider):

    def __init__(self,
                 seed = 43,
                 step_limit = 1,
                 volume_limit = 100,
                 lower_price_bound = 10,
                 walking_threshold = 0.5,
                 date_from: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 date_to: Optional[Union[date, datetime, arrow.Arrow]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        self.seed = seed
        np.random.seed(self.seed)


        self.walking_threshold = walking_threshold
        self.step_limit = step_limit
        self.volume_limit = volume_limit
        self.lower_price_bound = lower_price_bound
            
        if date_to is None:
            date_to = arrow.now()
        elif isinstance(date_to, date):
            date_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(date_to, datetime):
            date_to = arrow.get(date_to.astimezone(timezone.utc))
        assert isinstance(date_to, arrow.Arrow)
        self.date_to = date_to
        
        if date_from is None:
            date_from = date_to.shift(months=-12)
        elif isinstance(date_from, date):
            date_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
        if isinstance(date_from, datetime):
            date_from = arrow.get(date_from.astimezone(timezone.utc))
        assert isinstance(date_from, arrow.Arrow)
        self.date_from = date_from
        
        assert date_from <= date_to
        
        # Prepare episode variables
        self._datetime: Optional[datetime] = None
        self._last_price: Optional[float] = None
        self._episode_start_datetime: Optional[Union[datetime, arrow.Arrow]] = None
    
    def reset(self,
              episode_start_datetime: Union[None, arrow.Arrow, datetime] = None,
              episode_min_duration: Union[None, Real, timedelta] = None,
              rng: Optional[np.random.RandomState] = None,
              **kwargs
              ) -> datetime:
        
        # np.random.seed(self.seed)


        # Check episode_min_duration
        if episode_min_duration is None:
            episode_min_duration = timedelta(seconds=0)
        elif isinstance(episode_min_duration, Real):
            episode_min_duration = timedelta(seconds=float(episode_min_duration))
        elif isinstance(episode_min_duration, timedelta):
            pass
        else:
            raise ValueError('Invalid episode_min_duration value')
        assert episode_min_duration.total_seconds() >= 0

        # Check episode_start_datetime
        if episode_start_datetime is None:
            # Choose random datetime only if it is not set. Use self.seed to generate the date
            rt = np.random.random()
            s = self.date_from.timestamp()
            e = (self.date_to - episode_min_duration).timestamp()
            episode_start_datetime = s + rt * (e - s)
            episode_start_datetime = arrow.get(episode_start_datetime, tzinfo=timezone.utc).datetime
        elif isinstance(episode_start_datetime, datetime):
            pass
        elif isinstance(episode_start_datetime, arrow.Arrow):
            episode_start_datetime = episode_start_datetime.datetime
        else:
            raise ValueError('Invalid episode_start_datetime value')
        self._episode_start_datetime = episode_start_datetime


        # re-using datetime as seed for rng. Use also episode date as a salt, to ensure that every date is unique,
        #     but still reproducible and depends on self.seed
        datetime_arr = [episode_start_datetime.year, episode_start_datetime.month, episode_start_datetime.day,
                        self.seed]
        np.random.seed(datetime_arr)

        # print(f'Provider start datetime {episode_start_datetime} {datetime_arr}')
        # Generate random frequency

        self._datetime = episode_start_datetime
        self._last_price = 0
        
        return self._episode_start_datetime
    
    def __next__(self) -> NamedTuple:
        self._datetime += timedelta(seconds = 5 * np.random.random())
        t = (self._datetime - self._episode_start_datetime).total_seconds()
        direction = 1 if np.random.random() > self.walking_threshold else -1
        step = direction * self.step_limit * np.random.rand() 

        price = max(self.lower_price_bound, self._last_price + math.floor(1 * step)) # + self.noise_amplitude * noise

        bysell = np.random.random()
        operation = 'S' if (bysell > 0.5 ) else 'B'

        amount = np.random.randint(self.volume_limit) + 1
        self._last_price = price
        
        # print(f'_next {step} {price} {amount}')

        # Return next trade
        return Trade(
            datetime=self._datetime,
            operation=operation,
            amount=amount,
            price=price,
        )
    
    def close(self):
        self._datetime = None
        self._last_price = None
        self._episode_start_datetime = None

    @property
    def name(self) -> str:
        return 'RandomWalk'

    @property
    def session_start_datetime(self) -> Union[datetime, None]:
        return self.date_from.datetime

    @property
    def session_end_datetime(self) -> Union[datetime, None]:
        return self.date_to.datetime

    @property
    def episode_start_datetime(self) -> Union[datetime, None]:
        return self._episode_start_datetime
    
    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'step_limit=({self.step_limit}), '
            f'seed={self.seed}, '
            f'date_from={self.date_from}, date_to={self.date_to})'
        )
