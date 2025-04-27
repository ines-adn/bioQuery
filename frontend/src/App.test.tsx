import { render } from '@testing-library/react';
import '@testing-library/jest-dom';
import App from './App';

test('renders learn react link', () => {
  const { getByText } = render(<App />);
  expect(getByText(/learn react/i)).toBeInTheDocument();
});
