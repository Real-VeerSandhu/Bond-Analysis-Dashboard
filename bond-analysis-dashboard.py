import scipy.optimize as optimize
import streamlit as st
import numpy as np
import scipy.stats as stats
import plotly.express as px
import plotly.graph_objects as go

def calculate_ytm(fv, coup, T, price, freq=2, guess=0.05):
    freq = float(freq)
    periods = T * freq
    coupon = coup / 100 * fv / freq
    dt = [(i + 1) / freq for i in range(int(periods))]
    
    def ytm_func(y):
        return sum([coupon / (1 + y / freq) ** (freq * t) for t in dt]) + fv / (1 + y / freq) ** (freq * max(dt)) - price
    
    try:
        return optimize.brentq(ytm_func, -0.99, 1.0)  # Wider search range
    except ValueError:
        st.error("YTM calculation failed: No root found in the given range.")
        return None

def calculate_bond_price(face_value, coupon_rate, years, ytm):
    price = (coupon_rate * face_value * (1 - (1 + ytm) ** -years) / ytm) + face_value / (1 + ytm) ** years
    return price

def calculate_duration(face_value, coupon_rate, years, ytm, freq=2):
    coupon = face_value * coupon_rate / freq
    cash_flows = np.array([coupon] * int(years * freq) + [face_value])
    discount_factors = np.array([(1 + ytm / freq) ** -(i + 1) for i in range(len(cash_flows))])
    weighted_cash_flows = cash_flows * discount_factors
    durations = np.arange(1, len(cash_flows) + 1) / freq
    duration = np.sum(durations * weighted_cash_flows) / np.sum(weighted_cash_flows)
    return duration / freq

def calculate_modified_duration(duration, ytm, freq=2):
    return duration / (1 + ytm / freq)

def calculate_convexity(face_value, coupon_rate, years, ytm, freq=2):
    coupon = face_value * coupon_rate / freq
    cash_flows = np.array([coupon] * int(years * freq) + [face_value])
    discount_factors = np.array([(1 + ytm / freq) ** -(i + 1) for i in range(len(cash_flows))])
    weighted_cash_flows = cash_flows * discount_factors
    convexities = np.arange(1, len(cash_flows) + 1) * (np.arange(1, len(cash_flows) + 1) + 1)
    convexity = np.sum(convexities * weighted_cash_flows) / np.sum(weighted_cash_flows)
    return convexity / freq ** 2

def calculate_yield(face_value, coupon_rate, price):
    return (face_value * coupon_rate) / price

def monte_carlo_simulation2(face_value, coupon_rate, years, current_price, iterations=1000):
    prices = []
    for _ in range(iterations):
        ytm_sim = np.random.normal(loc=calculate_ytm(face_value, coupon_rate, years, current_price), scale=0.01)
        price_sim = calculate_bond_price(face_value, coupon_rate, years, ytm_sim)
        prices.append(price_sim)
    
    prices = np.array(prices)
    
    mean_price = np.mean(prices)
    std_dev = np.std(prices)
    p5 = np.percentile(prices, 5)
    p95 = np.percentile(prices, 95)
    min_price = np.min(prices)
    max_price = np.max(prices)
    
    stats_dict = {
        "mean_price": mean_price,
        "std_dev": std_dev,
        "p5": p5,
        "p95": p95,
        "min_price": min_price,
        "max_price": max_price
    }
    
    return prices, stats_dict

def monte_carlo_simulation(face_value, coupon_rate, years, current_price, iterations=1000):
    prices = []
    for _ in range(iterations):
        ytm_sim = np.random.normal(loc=calculate_ytm(face_value, coupon_rate, years, current_price), scale=0.01)
        price_sim = calculate_bond_price(face_value, coupon_rate, years, ytm_sim)
        prices.append(price_sim)
    return prices

st.set_page_config(
    page_title="Bond Analysis Dashboard",
    page_icon="logo.png",  
    layout="wide",  
)

st.title('Bond Analysis Dashboard')
st.sidebar.header('Bond Parameters')
face_value = st.sidebar.number_input('Face Value ($)', value=1000.00, format='%.2f')
coupon_rate = st.sidebar.number_input('Coupon Rate (%)', value=5.00, format='%.2f') / 100
years = st.sidebar.number_input('Years to Maturity', value=10.00, format='%.2f')
current_price = st.sidebar.number_input('Current Bond Price ($)', value=950.00, format='%.2f')
iterations = st.sidebar.slider('Monte Carlo Simulation Count', 5, 1000, 250, 1)
interest_shock = st.sidebar.slider('Interest Rate Shock (%)', -3.0, 3.0, 0.0, 0.1)


ytm = calculate_ytm(face_value, coupon_rate, years, current_price)
bond_price = calculate_bond_price(face_value, coupon_rate, years, ytm)
duration = calculate_duration(face_value, coupon_rate, years, ytm)
convexity = calculate_convexity(face_value, coupon_rate, years, ytm)
bond_yield = calculate_yield(face_value, coupon_rate, current_price)

ytms = np.linspace(0.01, 0.15, 50)
prices_curve = [calculate_bond_price(face_value, coupon_rate, years, ytm) for ytm in ytms]
fig_curve = go.Figure()
fig_curve.add_trace(go.Scatter(x=ytms * 100, y=prices_curve, mode='lines', name='Bond Price vs YTM'))
fig_curve.update_layout(title='Yield Curve Simulation', xaxis_title='Yield to Maturity (%)', yaxis_title='Bond Price ($)')


col1, col2 = st.columns([4,4])

with col1:
    st.plotly_chart(fig_curve)
    ytms_shocked = calculate_ytm(face_value, coupon_rate, years, current_price) + (interest_shock / 100)
    price_shocked = calculate_bond_price(face_value, coupon_rate, years, ytms_shocked)
    st.write(f'Impact of {interest_shock}% interest rate shock: `${price_shocked:.2f}`')
    st.write(f'Yield to Maturity (YTM): `{ytm:.6f}%`')
    st.write(f'Current Yield: ``{bond_yield:.2f}%``')
    st.write(f'Bond Price (calculated): `${bond_price:.2f}`')
    st.write(f'Bond Duration: ``{duration:.2f}``')
    st.write(f'Bond Convexity: ``{convexity:.2f}``')

with col2:
    prices, mc_stats = monte_carlo_simulation2(face_value, coupon_rate, years, current_price, iterations)

    title = f'Monte Carlo Simulation of Bond Price ({iterations} iterations)'
    fig = px.histogram(prices, nbins=100, title=title, color_discrete_sequence=['#4b64ff'])
    fig.update_layout(xaxis_title='Bond Price', yaxis_title='Frequency', showlegend=False)
    st.plotly_chart(fig)
    st.write(f'Monte Carlo Simulation Statistics ({iterations} iterations)')
    st.write(f'Mean Price: `${mc_stats["mean_price"]:.2f}`')
    st.write(f'Standard Deviation (Volatility): `${mc_stats["std_dev"]:.2f}`')
    st.write(f'5th Percentile (Lower Bound): `${mc_stats["p5"]:.2f}`')
    st.write(f'95th Percentile (Upper Bound): `${mc_stats["p95"]:.2f}`')
    st.write(f'Min Simulated Price: `${mc_stats["min_price"]:.2f}`')
    st.write(f'Max Simulated Price: `${mc_stats["max_price"]:.2f}`')

st.caption('Built with Python, Scipy, Numpy, Streamlit, Plotly')
st.caption('Veer Sandhu - 2025')
st.caption("[Github](https://github.com/Real-VeerSandhu/Bond-Analysis-Dashboard)")
