import pandas as pd



def clean_data(input):
    df = pd.read_csv(input)
    categories = df.Name.to_list()
    print(categories)



CATEGORIES = [
    ['Room', 'Bathroom', 'Bathroom Cleanliness', 'Bathroom Size', 'Beds', 'Balcony', 'View', 'Air Conditioning',
     'Minibar', 'TV', 'Furniture', 'Kitchenette', 'Family Room', 'Shower',
     'Room Cleanliness', 'Room Size', 'Room Maintenance', 'Old Room', 'New Room'],
    ['Comfort', 'Noise Level'],
    ['Cleanliness', 'Hotel Cleanliness'],
    ['Beach', 'Beach Cleanliness', 'Beach Sports'],
    ['Breakfast', 'Breakfast Prices'],
    ['Food', 'Food Prices', 'Dining Experience', 'Dining Area Cleanliness', 'Menu', 'Salads',
     'Meat', 'Beef Dishes', 'Steak', 'Pork Dishes', 'Poultry Dishes', 'Venison Dishes', 'Lamb Dishes',
     'Soups', 'Fish', 'Seafood','Side Dishes', 'Vegetables', 'Desserts & Fruits', 'Vegetarian & Vegan',
     'Pasta', 'Pizza', 'Sushi', 'Snacks'],
    ['Bar', 'Alcoholic Drinks', 'Bar Prices'],
    ['Location', 'Accessibility by car', 'Sightseeing', 'Restaurants & Bars', 'Shopping', 'Parking', 'Parking Prices',
     'Distance to City Centre', 'Distance to Public Transport', 'Distance to Train Station', 'Distance to Airport',
     'Distance to Business Sites', 'Distance to Winter Sports Facilities', 'Distance to Beach'],
    ['Service', 'Wellness Staff', 'Reception', 'Restaurant Service', 'Pool Service', 'Beach Service',
     'Housekeeping Staff',
     'Tour Guide', 'Management', 'Childcare', 'Bar Service', 'Laundry Service', 'Booking Process',
     'Hotel Security', 'Room Service', 'Romantic Decoration', 'Recreation Staff', 'Shuttle Service',
     'Valet Service', 'Concierge Service', 'Service Friendliness', 'Service Professionalism'],
    ['Vibe', 'Modern Vibe', 'Designer Vibe', 'Luxurious Vibe', 'Friendly Atmosphere'],
    ['WiFi', 'WiFi Cost', 'WiFi Quality'],
    ['Pool', 'Pool Cleanliness'],
    ['Wellness', 'Wellness Area Cleanliness', 'Sauna', 'Fitness Area', 'Fitness'],
    ['Value', 'Amenities', 'Grounds', 'Outdoor Sports Facilities', 'Kids Facilities', 'Golf Court',
     'Water Park', 'Hotel Building', 'Terrace', 'Entrance Area', 'Business Facilities', 'Smoking Area',
     'Ski Storage', 'Handicap Accessible Facilities', 'Shared Facilities', 'Architecture',
     'Casino Equipment', 'Elevator', 'Old Facilities', 'New Facilities'],
    ['Hotel', 'Luxury Hotel', 'Family Hotel', 'Romantic Hotel', 'Wellness Hotel', 'City Hotel',
     'Boutique Hotel', 'Business Hotel', 'Party Hotel', 'Economy Hotel', 'Golf Hotel', 'Resort Hotel',
     'Winter Sports Hotel', 'Beach Hotel', 'Solo Hotel', 'Eco-friendly Hotel', 'Casino Hotel',
     'Airport Hotel', 'Lake Hotel', 'Water Park Hotel', 'Highway Hotel', 'Hostel', 'Midscale Hotel',
     'Friends Hotel']
]


if __name__ == '__main__':
    inputfile = 'topic_modeling/categories_hotel_ty.csv'
    clean_data(inputfile)